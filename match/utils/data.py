from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.gaussian_blur import GaussianBlur
import torch 

import os
import glob
class ImageFolderData:
    def __init__(self):
        # setup dataTransform
        _color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
        _train_transforms = transforms.Compose([transforms.RandomResizedCrop(size = 224,scale=(0.5, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([_color_jitter], p = 0.7),
                                            transforms.RandomGrayscale(p  = 0.2),
                                            GaussianBlur(3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                            ])
        _val_transforms = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                         
        ])
        _test_transforms = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),  # i.e., no transformation
        ])

        self.transforms = {
            'train': _train_transforms,
            'val': _val_transforms, 
            'test': _test_transforms
        }

    def get_loader(self, batch_size, num_workers, shuffle=False, mode='train', img_dir=None, strict_lr_hr_correlation=False, drop_last=False):
        """
        Strict LR-HR correlation: If True, the dataloader will return a pair of images
        (LR, HR) for each image in the dataset. If False, the dataloader will return only
        the HR image.
        """
        assert img_dir is not None, "Please provide the path to your dataset."
        assert mode in ['train', 'val', 'test'], "Mode must be either 'train' or 'test'."

        apply_downsample_transform = (strict_lr_hr_correlation == True)

        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        dataset = ListedDataset(img_paths, self.transforms[mode], mode, apply_downsample_transform)
        loader  = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, 
                             drop_last=drop_last)
        return loader 


class ListedDataset(Dataset):
    def __init__(self, 
                 img_paths, 
                 transform, 
                 mode, 
                 size_hr: tuple=(224, 224),

                 apply_downsample_transform: bool=False, 
                 downsample_factor: int=4,
                 downsample_mode: Image.Resampling=Image.BICUBIC):
        """
        Args: 
          - img_paths: List of image file paths.
          - transform: Transform to apply to the images.
          - mode: 'train', 'val', or 'test'. 
          - size_hr: Size of the high-resolution images (default: (224, 224)).

          - apply_downsample_transform: If True, applies a downsample transform to the images.
          - downsample_factor: Factor by which to downsample the images (default: 4).
          - downsample_mode: Resampling mode for downsampling (default: Image.BICUBIC).
        """
        self.img_paths = img_paths
        self.transform = transform
        self.mode      = mode

        if apply_downsample_transform:
            __h, __w             = size_hr
            _downsample_factor   = downsample_factor
            self.prob_downsample = 0.5

            self.downsample_transform = transforms.Compose([
                transforms.Resize((__h//_downsample_factor, __w//_downsample_factor), interpolation=downsample_mode),
                transforms.Resize((__h, __w), interpolation=Image.BICUBIC),
            ])

        else:
            self.downsample_transform = None 

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img      = Image.open(img_path).convert('RGB')
        imgi     = self.transform(img)
        imgj     = self.transform(img)

        if self.downsample_transform is not None:
            if torch.rand(1) < self.prob_downsample:
                imgj = self.downsample_transform(imgj)
        if self.mode == 'test':
            imgj = None 
            return (imgi, img_path)
        return (imgi, imgj, img_path)

    def __len__(self):
        return len(self.img_paths)
