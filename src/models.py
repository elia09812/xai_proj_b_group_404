import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 1) Convolutional "feature extractor"
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 3, 128,128] -> [B,32,128,128]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,32,128,128] -> [B,32,64,64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,32,64,64]  -> [B,64,64,64]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,64,64,64]  -> [B,64,32,32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B,64,32,32]  -> [B,128,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,128,32,32] -> [B,128,16,16]
        )

        # 2) Linear "classifier" head
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # [B,128,32,32] -> [B, 128*32*32]
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class LargerNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout_p=0.5):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Globale Durchschnittspool-Lösung: funktioniert für jede Bildauflösung
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # FC + Dropout
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(128, num_classes)

        # Gemeinsamer Pool
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)             # (B,128,1,1)
        x = x.view(x.size(0), -1)   # (B,128)

        x = self.dropout(x)
        x = self.fc(x)
        return x