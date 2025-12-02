import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torch.utils import data as data

from .realesrgan import RealESRGAN_degradation
from .codeformer import Codeformer_degradation
from utils.img_util import convert_image_to_fn
from utils.misc import exists
import torch.nn.functional as F

from .basicsr.data.transforms import augment
Image.MAX_IMAGE_PIXELS = None

class LocalImageDataset(data.Dataset):
    def __init__(self, 
                gt_path="datasets/GT", 
                lr_path = 'datasets/LR',
                ref_path = 'datasets/Ref',
                text_path = 'datasets/texts',
                text_ref_path = 'datasets/ref_texts',
                image_size=512,
                tokenizer=None,
                accelerator=None,
                null_text_ratio=0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio

        self.text_path = text_path
        self.ref_text_path = text_ref_path

        #self.degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')
        self.degradation_realesrgan = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')
        self.degradation_codeformer = Codeformer_degradation('params_codeformer.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.img_paths = []
        self.img_paths.extend(sorted(glob.glob(f'{gt_path}/*.*g')[:]))
        self.lr_paths = []
        self.lr_paths.extend(sorted(glob.glob(f'{lr_path}/*.*g')[:]))
        self.ref_paths = []
        self.ref_paths.extend(sorted(glob.glob(f'{ref_path}/*.*g')[:]))


    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index].strip()

        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        lr = Image.open(self.lr_paths[index]).convert('RGB')
        ref = Image.open(self.ref_paths[index]).convert('RGB')
        image = np.asanyarray(image)/255
        lr = np.asanyarray(lr)/255
        ref = np.asanyarray(ref)/255

        image, lr, ref = augment([image, lr, ref], hflip=True)
        pixel_values = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
        ref = torch.from_numpy(ref).permute(2,0,1).float()
        example["ref_pixel_values"] = ref

        if random.random() < 0.9:
            conditioning_pixel_values = torch.from_numpy(lr).permute(2,0,1).unsqueeze(0).float()
            
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            ori_h, ori_w = pixel_values.shape[2], pixel_values.shape[3]
            conditioning_pixel_values = F.interpolate(conditioning_pixel_values, size=(ori_h, ori_w), mode=mode)
            
            example["pixel_values"] = pixel_values.squeeze(0).mul(2).sub(1.0)
            example["conditioning_pixel_values"] = conditioning_pixel_values.squeeze(0)
        else:
            degradation = self.degradation_realesrgan
            GT_image_t, LR_image_t = degradation.degrade_process(pixel_values, use_large=True, resize_bak=self.resize_bak)
            example["conditioning_pixel_values"] = LR_image_t.squeeze(0)
            example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0 
                       

        caption = ""
        txt_path = os.path.join(self.text_path, img_path.split('/')[-1].split('_')[-2] + '.txt')
        if os.path.exists(txt_path):
            fp = open(txt_path, "r")
            caption = fp.readlines()[0]
            fp.close()      

        ref_txt_path = os.path.join(self.ref_text_path, os.path.basename(img_path).split('.')[-2]+'.txt')
        if os.path.exists(ref_txt_path):
            fp = open(ref_txt_path, "r")
            ref_caption = fp.readlines()[0]
            fp.close()

        return example["pixel_values"], caption, ref_caption, "", example["conditioning_pixel_values"], example["ref_pixel_values"]

    def __len__(self):
        return len(self.img_paths)
