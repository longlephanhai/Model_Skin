import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
from torch.utils.data import Dataset, DataLoader


class SkinDiseaseDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = ['blackheads', 'dark spot',
                           'nodules', 'papules', 'pustules', 'whiteheads']
        self.transform = transform
        data_path = os.path.normpath(os.path.join(
            root))

        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "valid")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        # image.show(title=label)
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    dataset = SkinDiseaseDataset(
        root="./dataset_classification", train=True, transform=transform)
    image, label = dataset.__getitem__(2)
    # print(image.shape, label)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    for images, labels in train_dataloader:
        print(images.shape, labels)
        break
