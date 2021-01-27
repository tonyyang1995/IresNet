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
#from PIL import Image
import pydicom



class OriginDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_paths, self.labels = self.__get_paths(opt.dataroot)
        print(len(self.img_paths), len(self.labels))
        #self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_paths = self.img_paths[index]
        #print(len(img_paths))
        label = self.labels[index]
        imgs = None

        np_array = None
        for i in range(0, len(img_paths)):
            img_path = img_paths[i]
            dcm = pydicom.dcmread(img_path, force=True)
            dcm_np = dcm.pixel_array
            dcm_np = dcm_np[np.newaxis,np.newaxis,:]
            if np_array is None:
                np_array = dcm_np
            else:
                np_array = np.concatenate((np_array, dcm_np))

        imgs = torch.from_numpy(np_array).float()
        imgs = imgs[:,:, 10:186, :] 
        return imgs, label, img_paths

    def name(self):
        return "OriginDataset"

    def __get_paths(self, dataroot):
        # the format is like img_paths['the patient'][all imgs]
        img_paths = []
        labels = []
        for r, dirs, files in os.walk(dataroot):
            patient_imgs = []
            ASDP = False
            for img in files:
                if not '.dcm' in img:
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
