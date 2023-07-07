from typing import Any
import torch, os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class PokeData(Dataset):
    def __init__(self,folder,tarsize=(56,56)):
        self.imglink = []
        for file in os.listdir(folder):
            if(file.endswith('.png')):
                self.imglink.append(os.path.join(folder,file))
        
        self.length = len(self.imglink)

        print(f'Loaded {self.length} images')
        self.transfo = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(tarsize)
        ])
    
    def __getitem__(self, index) -> torch.Tensor:
        link = self.imglink[index]

        img=self.transfo(Image.open(link).convert('RGBA'))

        return img

    def __len__(self):
        return self.length