import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models
from torch import nn
from typing import Tuple


LABELS_MAP = {
    "coffee-mug": 0,
    "notebook": 1,
    "remote-control": 2,
    "soup-bowl": 3,
    "teapot": 4,
    "wooden-spoon": 5,
    "computer-keyboard": 6,
    "mouse": 7,
    "binder": 8,
    "toilet-tissue": 9,
}

IDX_TO_CLASS = {v: k for k, v in LABELS_MAP.items()}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImageDataset(Dataset):
    def __init__(self, root: str, transform=None, train: bool = True):
        self.train = train
        self.root = root

        self.file_names = sorted(
            f for f in os.listdir(self.root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        self.transform = transform

    def __getitem__(self, idx: int):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.root, img_name)

        if self.train:
            label_str = img_name.split("_")[0]
        else:
            label_str = img_name.split("_")[2]

        label = torch.tensor(LABELS_MAP[label_str], dtype=torch.long)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_name

    def __len__(self):
        return len(self.file_names)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(131072, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    import torch.nn.functional as F

class LargerNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout_p=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x



def build_resnet18_for_10_classes():
  
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    model = models.resnet18(weights=weights, progress=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
 
    return model, preprocess

def build_simplenet_for_10_classes():

    model = SimpleCNN(num_classes=10)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return model, preprocess

def build_largernet_for_10_classes():
   
    model = LargerNet(num_classes=10)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return model, preprocess



def build_model(model_name: str) -> Tuple[nn.Module, object]:

    name = model_name.lower()

    if name == "resnet18":
        return build_resnet18_for_10_classes()

    if name == "simplenet":
        return build_simplenet_for_10_classes()

    if name == "largernet":
        return build_largernet_for_10_classes()

    raise ValueError(f"Unknown model_name: {model_name}")




