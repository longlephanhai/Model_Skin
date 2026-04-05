import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torchsummary import summary 


class SkinDiseaseModelResNet(nn.Module):
  def __init__(self, num_classes=6):
    super().__init__()
    self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1024),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(in_features=1024, out_features=num_classes)
    )

    for name, param in self.model.named_parameters():
        if 'fc.' in name or "layer4." in name:
            pass
        else:
            param.requires_grad = False
        # print(f"{name}: requires_grad={param.requires_grad}")
    
  def forward(self, x):
    return self.model(x)
  
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = SkinDiseaseModelResNet(num_classes=6).to(device)
    
#     summary(model, (3, 224, 224))