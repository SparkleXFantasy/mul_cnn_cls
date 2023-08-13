import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FolderDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(FolderDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.supported = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp']
        self.classes, self.class_to_idx = self.__find_classes(data_root)
        self.images = self.__make_dataset(data_root, self.classes, self.class_to_idx)

    def __find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def __make_dataset(self, dir, classes, class_to_idx):
        images = []
        for cls in classes:
            for root, _, names in os.walk(os.path.join(dir,cls)):
                for name in names:
                    if os.path.splitext(name)[-1].lower() in self.supported:
                        images.append((os.path.join(root, name), class_to_idx[cls]))
        return images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_pil = Image.open(self.images[idx][0]).convert('RGB')
        img_cls = torch.tensor(self.images[idx][1], dtype=torch.int64)
        img_transforms = self.transform if self.transform else transforms.Compose([
            transforms.ToTensor(),
        ])
        img_t = img_transforms(img_pil)
        return img_t, img_cls
