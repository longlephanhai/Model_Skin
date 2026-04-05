from torchvision.transforms import Compose, RandomAffine, Resize, ToTensor, Normalize, ColorJitter
from dataset_classification import SkinDiseaseDataset
from torch.utils.data import DataLoader
import cv2
import torch.optim
import numpy as np
# from model import SkinDiseaseModel
from skin_disease_model_resnet import SkinDiseaseModelResNet
import torch.nn as nn
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import shutil
import os


def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str,
                        default="./dataset_classification", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int,
                        default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int,
                        default=16, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int,
                        default=224, help="Image size"),
    parser.add_argument("--nums-classes", "-nc", type=int,
                        default=6, help="Number of classes"),
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t",
                        type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.05, 0.05),
            scale=(0.85, 1.15),
            shear=10,
        ),
        ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        # Resize((224, 224)),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SkinDiseaseDataset(
        root="./dataset_classification", train=True, transform=train_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        # batch_size=4,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # image, label = train_dataset.__getitem__(1)
    # image = (torch.permute(image, (1, 2, 0))*255.0).numpy().astype(np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Test image", image)
    # cv2.waitKey(0)

    test_transform = Compose([
        # Resize((224, 224)),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SkinDiseaseDataset(
        root="./dataset_classification", train=False, transform=test_transform)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        # batch_size=4,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    writer = SummaryWriter(args.logging)
    # model = SkinDiseaseModel(num_classes=22)
    model = SkinDiseaseModelResNet(num_classes=args.nums_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5)

    scaler = torch.amp.GradScaler()

    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=True)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        start_epoch = 0
        best_accuracy = 0

    # epochs = 10
    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):
        # Train Phase
        model.train()
        progress_bar = tqdm(train_dataloader, colour="red",
                            desc="Epoch {}".format(epoch+1))
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # forward
            with torch.amp.autocast(device_type=device.type):
                predicts = model(images)
                loss = criterion(predicts, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(
                epoch+1, args.epochs, iter+1, num_iters, loss))
            writer.add_scalar("Train/Loss", loss, epoch*num_iters+iter)
            # backward

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # Validation Phase
        model.eval()
        all_predictions = []
        all_labels = []
        val_progress = tqdm(test_dataloader, colour="green",
                            desc="Validating", leave=False)
        for iter, (images, labels) in enumerate(val_progress):
            images = images.to(device)
            labels = labels.to(device)
            all_labels.extend(labels.detach().cpu().numpy())
            # print("all labels ", all_labels)
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions, dim=1).cpu()
                all_predictions.extend(indices)
                loss = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        scheduler.step(accuracy)
        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)

        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "best_accuracy": best_accuracy
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_accuracy:
            checkpoint = {
                "epoch": epoch+1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_accuracy": accuracy
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_accuracy = accuracy
