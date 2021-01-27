import random
import numpy as np
import torch.utils.data as data
#from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

from .Base_dataset import BaseDataset

import os
import pydicom as dicom
from PIL import Image


class CubeDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_paths, self.labels = self.__get_paths(opt.dataroot)
        self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_paths = self.img_paths[index]
        #print(len(img_paths))
        label = self.labels[index]
        imgs = None

        for i in range(5, 150):
            img_path = img_paths[i]
            img = Image.open(img_path).convert('RGB')
            #img = img.resize((128,128))
            #img = img.crop((16, 16, 224 + 16, 224 + 16))
            # extract the brain, get rid of other parts
            img = img.crop((10, 0, 186, 256))
            # just transfer to tensor
            img = self.transform(img)
            imgs = img.unsqueeze(0) if imgs is None else torch.cat([imgs, img.unsqueeze(0)], dim=0)	
        #print(imgs.size())
        return imgs, label, img_paths

    def name(self):
        return "CubeDataset"

    def __get_paths(self, dataroot):
        # the format is like img_paths['the patient'][all imgs]
        img_paths = []
        labels = []
        for r, dirs, files in os.walk(dataroot):
            patient_imgs = []
            ASDP = False
            for img in files:
                if not '.png' in img:
                    continue
                patient_imgs.append(os.path.join(r, img))
                if not ASDP and 'ASDP' in r:
                    ASDP = True
            if (len(patient_imgs) == 0):
                continue
            patient_imgs = sorted(patient_imgs)
            #print(len(patient_imgs))
            img_paths.append(patient_imgs)                  
            if ASDP:
                labels.append(1)
            else:
                labels.append(0)
                                
        return img_paths, labels
