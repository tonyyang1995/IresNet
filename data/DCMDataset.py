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


class DCMDataset(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths, self.labels = self.__get_paths(opt.dataroot)
		self.transform = transforms.ToTensor()

		#self.split_tensor()
		#print("already split the tensors")
		#print(len(self.img_paths), len(self.labels))

	def __len__(self):
		return len(self.img_paths)
		#return len(self.cubes)

	def __getitem__(self, index):
		img_paths = self.img_paths[index]
		label = self.labels[index]
		imgs = None
		for img_path in img_paths:
			img = Image.open(img_path).convert('RGB')
			img = img.resize((256,256))
			# just transfer to tensor
			img = self.transform(img)
			imgs = img.unsqueeze(0) if imgs is None else torch.cat([imgs, img.unsqueeze(0)], dim=0)	
		# start_index = random.randint(0, len(img_paths)- self.opt.N)
		# crop_h = random.randint(0, 256 - self.opt.h)
		# crop_w = random.randint(0, 256 - self.opt.w)
		# #for i in range(start_index, start_index + self.opt.N):
		# for i in range(0, len(img_paths)):
		# 	img = Image.open(img_paths[i]).convert('RGB')
		# 	img = img.crop((crop_h, crop_w, crop_h+self.opt.h, crop_w+self.opt.w))
		# 	img = self.transform(img)
		# 	imgs = img.unsqueeze(0) if imgs is None else torch.cat([imgs, img.unsqueeze(0)], dim=0)

		return imgs, label, img_paths

	def name(self):
		return "DCMDataset"

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

class DCMtestDataset(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths = self.__get_paths(opt.dataroot)
		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.img_paths)

	def name(self):
		return 'DCMtestDataset'

	# def split_tensor(self):
	# 	self.cubes = []
	# 	self.cube_labels = []
	# 	self.names = []
	# 	for i, img_paths in enumerate(self.img_paths):
	# 		label = self.labels[i]
	# 		imgs = None

	# 		for img_path in img_paths:
	# 			img = Image.open(img_path).convert('RGB')
	# 			img = img.resize((256, 256))
	# 			img = self.transform(img)
	# 			imgs = img.unsqueeze(0) if imgs is None else torch.cat([imgs, img.unsqueeze(0)], dim=0)

	# 		_, Dim, Height, Width = imgs.size()
	# 		# split
	# 		for h in range(0, Height-128, 16):
	# 			for w in range(0, Width-128, 16):
	# 				cube = imgs[:, :, h:h+128, w:w+128]
	# 				self.cubes.append(cube)
	# 				self.cube_labels.append(label)
	# 				self.names.append(img_paths)

	def __getitem__(self, index):
		img_paths = self.img_paths[index]
		#label = self.labels[index]
		imgs = None
		for img_path in img_paths:
			img = Image.open(img_path).convert('RGB')
			# just transfer to tensor
			img = self.transform(img)
			imgs = img.unsqueeze(0) if imgs is None else torch.cat([imgs, img.unsqueeze(0)], dim=0)
	
		return imgs, img_paths

	def __get_paths(self, dataroot):
		# the format is like img_paths['the patient'][all imgs]
		img_paths = []
		for r, dirs, files in os.walk(dataroot):
			patient_imgs = []
			for img in files:
				if not '.png' in img:
					continue
				patient_imgs.append(os.path.join(r, img))

			if (len(patient_imgs) == 0):
				continue
			patient_imgs = sorted(patient_imgs)
			img_paths.append(patient_imgs)
				
		return img_paths