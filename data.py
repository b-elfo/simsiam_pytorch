import os

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

###

class SimSiamDataset(Dataset):
    def __init__(self, 
                 path):
        self.path = path
        self.img_dirs = [ img_dir for img_dir in os.listdir(path) if '.jpg' in img_dir ]
        self.transform = self.simsiam_transforms()

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, 
                     idx):
        img_dir = os.path.join( self.path, self.img_dirs[idx] )
        img = Image.open(img_dir)
        target1 = self.transform(img)
        target2 = self.transform(img)
        return target1, target2

    def simsiam_transforms(self, 
                           size: int = 256):
        """
        
        """
        # crop
        crop_scale = (0.5,1.0)
        # flip
        flip_p = 0.5
        # colour
        brightness = 0.3
        contrast = 0.3
        saturation = 0.3
        hue = 0.15
        colour_p = 0.8
        # gray
        gray_p = 0.2
        # gaussian
        kernel = int(0.1*256)
        kernel += (kernel-1)%2
        sigma = (0.1,2.0)
        gauss_p = 0.5
        return transforms.Compose([
                                   transforms.RandomResizedCrop(size, scale=crop_scale),
                                   transforms.RandomHorizontalFlip(p=flip_p),
                                   transforms.RandomApply([transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)], p=colour_p),
                                   transforms.RandomGrayscale(p=gray_p),
                                   transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)], p=gauss_p),
                                   transforms.ToTensor()
                                   ])

def dataloader(path,
               batch_size=32, 
               shuffle=True, 
               num_workers=1,
               ):
    """
    
    """
    dataset = SimSiamDataset(path)
    valid_size = dataset.__len__()//4
    data_split = [dataset.__len__()-valid_size, valid_size]
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, 
                                                                 lengths=data_split)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  )
    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  )
    return train_dataloader, valid_dataloader
