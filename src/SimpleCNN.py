class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Flatten(),                                # [B,128,16,16] -> [B, 128*16*16]
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x