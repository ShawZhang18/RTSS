import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import cv2
import imageio

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path).convert('L')

def disparity_loader(path):
    return Image.open(path)

def semantic_gt_loader_np(path):
    return np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

# def semantic_gt_loader_np(path):
#     return imageio.imread(path)

w_crop = 1232
h_crop = 368

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, semantic, training, fn=None, loader=default_loader, semantic_loader = semantic_gt_loader_np, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.semantic = semantic
        self.loader = loader
        self.dploader = dploader
        self.semantic_loader = semantic_loader
        self.training = training
        self.fn = fn

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]
        semantic_gt = self.semantic[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        semantic_img = self.semantic_loader(semantic_gt)
        dataL = self.dploader(disp_L)

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
           # semantic_img = semantic_img.crop((x1, y1, x1 + tw, y1 + th))
           # semantic_img = transforms.ToTensor()(semantic_img)
           # semantic_img = semantic_img.long()
           semantic_img = semantic_img[y1:y1+th, x1:x1+tw]

           dataL = np.ascontiguousarray(dataL,dtype=np.float32) / 256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL, semantic_img
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-w_crop, h-h_crop, w, h))
           right_img = right_img.crop((w-w_crop, h-h_crop, w, h))
           semantic_img = semantic_img[h-h_crop:h, w-w_crop:w]

           dataL = dataL.crop((w-w_crop, h-h_crop, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32) / 256

           processed = preprocess.get_transform(augment=False)
           left_img       = processed(left_img)
           right_img      = processed(right_img)
           fn0 = self.fn[index]

           return left_img, right_img, dataL, semantic_img, fn0

    def __len__(self):
        return len(self.left)
