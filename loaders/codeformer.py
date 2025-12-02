import os
import copy
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from .basicsr.data.transforms import augment
from .basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from .basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from .basicsr.utils.img_process_util import filter2D
from .basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)

cur_path = os.path.dirname(os.path.abspath(__file__))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader) 

    return opt

class Codeformer_degradation(object):
    def __init__(self, opt_name='params_codeformer.yml', device='cpu'):
        opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        self.device = device #torch.device('cpu')

        # perform corrupt
        self.use_corrupt = self.opt.get('use_corrupt', True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_motion_kernel:
            self.motion_kernel_prob = self.opt.get('motion_kernel_prob', 0.001)
            motion_kernel_path = self.opt.get('motion_kernel_path', 'datasets/motion-blur-kernels-32.pth')
            self.motion_kernels = torch.load(motion_kernel_path)

        if self.use_corrupt:
            # degradation configurations
            self.blur_kernel_size = self.opt['blur_kernel_size']
            self.kernel_list = self.opt['kernel_list']
            self.kernel_prob = self.opt['kernel_prob']
            # Small degradation
            self.blur_sigma = self.opt['blur_sigma']
            self.downsample_range = self.opt['downsample_range']
            self.noise_range = self.opt['noise_range']
            self.jpeg_range = self.opt['jpeg_range']
            # Large degradation
            self.blur_sigma_large = self.opt['blur_sigma_large']
            self.downsample_range_large = self.opt['downsample_range_large']
            self.noise_range_large = self.opt['noise_range_large']
            self.jpeg_range_large = self.opt['jpeg_range_large']

            # print
            print(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
            print(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
            print(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            print(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = self.opt.get('color_jitter_prob', None)
        self.color_jitter_pt_prob = self.opt.get('color_jitter_pt_prob', None)
        self.color_jitter_shift = self.opt.get('color_jitter_shift', 20)
        if self.color_jitter_prob is not None:
            print(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')

        # to gray
        self.gray_prob = self.opt.get('gray_prob', 0.0)
        if self.gray_prob is not None:
            print(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.
    
    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def random_augment(self, img_gt):
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
        """
        # random color jitter 
        if np.random.uniform() < self.opt['color_jitter_prob']:
            jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.opt['gray_prob']:
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)

        return img_gt

    @torch.no_grad()
    def degrade_process(self, img_gt, use_large=True, resize_bak=True):
        img_gt = self.random_augment(img_gt)
        ori_h, ori_w = img_gt.shape[:2]
        
        img_in = img_gt
        if self.use_corrupt:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_in = cv2.filter2D(img_in,-1,k)
                
            # gaussian blur
            #self.kernel_range = [2 * v + 1 for v in range(15, 20)] # 31-41
            # self.kernel_range = [2 * v + 1 for v in range(15, 25)] # 31-51
            # blur_kernel_size = random.choice(self.kernel_range) if False else self.blur_kernel_size #
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma_large if use_large else self.blur_sigma,
                self.blur_sigma_large if use_large else self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None)
            img_in = cv2.filter2D(img_in, -1, kernel)

            # downsample
            scale = np.random.uniform(self.downsample_range_large[0], self.downsample_range_large[1]) if use_large else np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_in = cv2.resize(img_in, (int(ori_h // scale), int(ori_w // scale)), interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range is not None:
                noise_sigma = np.random.uniform(self.noise_range_large[0] / 255., self.noise_range_large[1] / 255.) if use_large else np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
                img_in = img_in + noise
                img_in = np.clip(img_in, 0, 1)

            # jpeg
            if self.jpeg_range is not None:
                jpeg_p = np.random.uniform(self.jpeg_range_large[0], self.jpeg_range_large[1]) if use_large else np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_p)]
                _, encimg = cv2.imencode('.jpg', img_in * 255., encode_param)
                img_in = np.float32(cv2.imdecode(encimg, 1)) / 255.

            # resize to in_size
            #img_in = cv2.resize(img_in, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_in = self.color_jitter(img_in, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            img_in = np.tile(img_in[:, :, None], [1, 1, 3])
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=False, float32=True)
        img_in, img_gt = img_in.unsqueeze(0), img_gt.unsqueeze(0)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_in = self.color_jitter_pt(img_in, brightness, contrast, saturation, hue)

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            img_in = F.interpolate(img_in, size=(ori_h, ori_w), mode=mode)

        # clamp and round
        img_lq = torch.clamp((img_in * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lq


