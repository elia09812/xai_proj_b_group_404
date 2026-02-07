import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models
from torch import nn
from typing import Tuple


# Mapping von Klassenname → Klassenindex
# Das ist exakt das gleiche Mapping wie im Training,
# damit Vorhersagen und Ground Truth vergleichbar bleiben.
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

# Um später von Index wieder auf Klassenname zu kommen (für Plots / Reports)
IDX_TO_CLASS = {v: k for k, v in LABELS_MAP.items()}


def get_device() -> torch.device:
    """
    Ich wähle automatisch das beste verfügbare Device:
    - CUDA, falls vorhanden (NVIDIA GPU)
    - sonst MPS auf dem Mac (Apple Silicon)
    - sonst CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImageDataset(Dataset):
    """
    Dataset-Klasse, identisch zur Trainingslogik.
    Für das Testen gebe ich zusätzlich den Dateinamen zurück,
    damit ich später pro Bild Vorhersagen analysieren kann.
    """
    def __init__(self, root: str, transform=None, train: bool = True):
        self.train = train
        self.root = root

        # Alle Bilddateien im Ordner sammeln
        self.file_names = sorted(
            f for f in os.listdir(self.root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        self.transform = transform

    def __getitem__(self, idx: int):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.root, img_name)

        # Label aus dem Dateinamen extrahieren
        # (gleiches Schema wie im Training)
        if self.train:
            label_str = img_name.split("_")[0]
        else:
            label_str = img_name.split("_")[2]

        label = torch.tensor(LABELS_MAP[label_str], dtype=torch.long)

        # Bild laden und Transform anwenden
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Fürs Testen gebe ich auch den Dateinamen zurück
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
    """
    Ich baue hier exakt das gleiche ResNet18 wie im Training:
    - Vortrainierte ImageNet-Gewichte
    - Letzte Fully-Connected-Schicht auf 10 Klassen geändert
    - Die ImageNet-Transforms werden mit zurückgegeben,
      damit Training und Test identisch vorverarbeitet werden.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    model = models.resnet18(weights=weights, progress=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
 
    return model, preprocess

def build_simplenet_for_10_classes():
    """
    Ich baue hier mein SimpleCNN und nutze für Tests dieselben
    Preprocess-Transforms wie im SimpleCNN-Training.
    """
    # NOTE: wenn SimpleCNN-Klasse in ml_shared steht, einfach verwenden:
    model = SimpleCNN(num_classes=10)

    # SimpleCNN hat keine festen "ImageNet weights.transforms()".
    # Ich setze hier die gleiche Eval-Transform wie in deinem SimpleCNN-Notebook:
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return model, preprocess

def build_largernet_for_10_classes():
    """
    Ich baue hier LargerNet für 10 Klassen.
    Als Preprocess nutze ich den gleichen Standard wie im SimpleCNN-Test:
    Resize 256x256 + Normalize auf mean/std 0.5.
    (Falls ihr beim LargerNet andere Transforms hattet, müssen wir das angleichen.)
    """
    model = LargerNet(num_classes=10)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return model, preprocess




# Ich nutze hier einen zentralen "Model Builder", damit meine Testskripte später
# einfach model = build_model("resnet18") machen können.
def build_model(model_name: str) -> Tuple[nn.Module, object]:
    """
    Gibt (model, preprocess) zurück.
    preprocess ist die Transform-Pipeline, die ich beim Testen anwenden muss,
    damit es exakt wie im Training ist.
    """
    name = model_name.lower()

    if name == "resnet18":
        return build_resnet18_for_10_classes()

    if name == "simplenet":
        # TODO: wenn du SimpleCNN schon in ml_shared hast, hier zurückgeben.
        # Falls nicht, kann ich dir gleich eine build_simplenet() hinzufügen.
        return build_simplenet_for_10_classes()

    if name == "largernet":
        # TODO: das ist der Platzhalter für eure LargerNet-Architektur.
        return build_largernet_for_10_classes()

    raise ValueError(f"Unknown model_name: {model_name}")




