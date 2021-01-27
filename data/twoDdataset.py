import random
import numpy as np
import torch.utils.data as data

from .Base_dataset import BaseDataset

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
from PIL import Image

class testDataset(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths = self.__get_paths(opt.dataroot)
		self.transform = transforms.ToTensor()


	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		img_paths = self.img_paths[index]
		imgs = []
		#print(len(img_paths))
		if 'ASDP' in img_paths[0]:
			labels = torch.tensor([1])
		else:
			labels = torch.tensor([0])

		for path in img_paths:
			img = Image.open(path).convert('RGB')
			img = img.resize((256, 256))
			img = self.transform(img)
			imgs.append(img)
		return imgs, labels, img_paths

	def name(self):
		return 'testDataset'

	def __get_paths(self, dataroot):
		img_paths = []
		for r, dirs, files in os.walk(dataroot):
			patient_imgs = []
			for img in files:
				if not '.png' in img:
					continue
				patient_imgs.append(os.path.join(r, img))
			patient_imgs = sorted(patient_imgs)
			#print(len(patient_imgs))
			if len(patient_imgs) > 0:
				img_paths.append(patient_imgs)
		return img_paths

class twoDDataset(BaseDataset):
	def __init__(self, opt):
		#super(twoDDataset, self).__init__()
		self.opt = opt
		self.image_paths, self.labels = self.__get_paths(opt.dataroot)
		#print(len(self.image_paths), len(self.labels))
		#print(len(self.img_paths), len(self.labels))
		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image, label = self.image_paths[index], self.labels[index]
		#print(image)
		img = Image.open(image).convert('RGB')
		img = img.resize((256,256))
		img = self.transform(img)
		#label = label.unsqueeze(0)
		#print(img.size())
		return img, label

	def __get_paths(self, dataroot):
		image_paths = []
		labels = []
		for r, dirs, files in os.walk(dataroot):
			for img in files:
				if not '.png' in img:
					continue
				image_paths.append(os.path.join(r, img))
				if 'ASDP' in r:
					labels.append(torch.tensor([1]).long())
				else:
					labels.append(torch.tensor([0]).long())
		return image_paths, labels

	def name(self):
		return 'twoDDataset'