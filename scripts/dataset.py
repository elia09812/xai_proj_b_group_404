"""
dataset.py
----------
Custom dataset class for loading images from directories where each subfolder corresponds to a class label.

"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
                            Should contain one subdirectory per class.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Alle (Klassenname, Bildpfad)-Paare auflisten
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            self.class_to_idx[class_name] = idx
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
from torch.utils.data import DataLoader
from torchvision import transforms

# Transformationen definieren (z. B. fürs Training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset laden
train_dataset = CustomImageDataset(root_dir="data/ImageNetSubset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Testlauf
images, labels = next(iter(train_loader))
print(f"Batchgröße: {images.shape}, Labels: {labels[:5]}")

# angenommen train_dataset ist dein CustomImageDataset
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
print("class_to_idx:", train_dataset.class_to_idx)
print("erste 5 label-indices:", labels[:5])
print("erste 5 class-names:", [idx_to_class[int(i)] for i in labels[:5]])

print(images.shape)   # z.B. torch.Size([32, 3, 64, 64])
print(labels.shape)   # z.B. torch.Size([32])
print(labels.dtype)   # z.B. torch.int64