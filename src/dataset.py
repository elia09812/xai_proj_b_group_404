
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.root = root
        self.file_names = sorted(
            f for f in os.listdir(self.root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        self.labels_map = {
            'coffee-mug': 0,
            'notebook': 1,
            'remote-control': 2,
            'soup-bowl': 3,
            'teapot': 4,
            'wooden-spoon': 5,
            'computer-keyboard': 6,
            'mouse': 7,
            'binder': 8,
            'toilet-tissue': 9,
        }
        self.transform = transform
        
    
    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.root, img_name)
        
        if self.train:
            label_str = img_name.split('_')[0]
        else:
            label_str = img_name.split('_')[2]
        label = torch.tensor(self.labels_map[label_str], dtype=torch.long)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.file_names)