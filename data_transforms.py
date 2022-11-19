import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image, ImageOps, ImageFilter
import random

def add_normalization_to_transform(unnormalized_transforms):
    """Adds ImageNet normalization to all transforms"""
    normalized_transform = {}
    for key, value in unnormalized_transforms.items():
        normalized_transform[key] = transforms.Compose([value, 
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                             std=[0.229, 0.224, 0.225])]) 
    return normalized_transform

def modify_transforms(normal_transforms, no_shift_transforms, ig_transforms):
    normal_transforms = add_normalization_to_transform(normal_transforms)
    no_shift_transforms = add_normalization_to_transform(no_shift_transforms)
    ig_transforms = add_normalization_to_transform(ig_transforms)
    return normal_transforms, no_shift_transforms, ig_transforms
    
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
# no imagent normalization for simclrv2
pure_transform = transforms.Compose([transforms.Resize(256), 
                                     transforms.CenterCrop(224), 
                                     transforms.ToTensor()])   

aug_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(21,21), sigma=(0.1,2.0))], p=0.5),
                                    transforms.ToTensor()])  

ig_pure_transform = transforms.Compose([transforms.Resize(256), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor()])   

ig_transform_colorjitter = transforms.Compose([transforms.Resize(256), 
                                               transforms.CenterCrop(224),
                                               transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)], p=1),
                                               transforms.ToTensor()])  

ig_transform_blur = transforms.Compose([transforms.Resize(256), 
                                        transforms.CenterCrop(224),
                                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(11,11), sigma=(5,5))], p=1),
                                        transforms.ToTensor()])  

ig_transform_solarize = transforms.Compose([transforms.Resize(256), 
                                            transforms.CenterCrop(224),
                                            Solarization(p=1.0),
                                            transforms.ToTensor()]) 

ig_transform_grayscale = transforms.Compose([transforms.Resize(256), 
                                             transforms.CenterCrop(224),
                                             transforms.RandomGrayscale(p=1),
                                             transforms.ToTensor()])  


ig_transform_combine = transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224),
                                           transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                           transforms.RandomGrayscale(p=0.2),
                                           transforms.RandomApply([transforms.GaussianBlur(kernel_size=(21,21), sigma=(0.1, 2.0))], p=0.5),
                                           transforms.ToTensor()])  

pure_transform_no_shift = transforms.Compose([transforms.Resize((224, 224)), 
                                              transforms.ToTensor()])   

aug_transform_no_shift = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                             transforms.RandomGrayscale(p=0.2),
                                             transforms.ToTensor()])  

normal_transforms = {'pure': pure_transform, 
                     'aug': aug_transform}

no_shift_transforms = {'pure': pure_transform_no_shift, 
                       'aug': aug_transform_no_shift}

ig_transforms = {'pure': ig_pure_transform, 
                 'color_jitter': ig_transform_colorjitter, 
                 'blur': ig_transform_blur, 
                 'grayscale': ig_transform_grayscale, 
                 'solarize': ig_transform_solarize,
                 'combine': ig_transform_combine}