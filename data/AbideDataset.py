import os
import nibabel as nib
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import numpy as np 

from .Base_dataset import BaseDataset
from .resample3d import resample3d


class OursDataset(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		# read the paths and labes from txt
		self.img_paths, self.labels = self.__get_from_txt(opt.dataroot)
		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		img_paths = self.img_paths[index]
		label = self.labels[index]
		imgs = nib.load(img_paths)
		#(img_paths)
		imgs = imgs.get_data()
		imgs = torch.from_numpy(imgs)

		if (len(imgs.size()) == 3):
			imgs = imgs.unsqueeze(3)

		if 'ASD' in img_paths:
			imgs = imgs.float().permute(3,0,1,2)
			imgs[:, :, 0:10, :] = 0
			imgs[:, :, 187:, :] = 0
		else:
			imgs = imgs.float().unsqueeze(0).unsqueeze(0)
			s = self.opt.size
			imgs = F.interpolate(imgs, size=s, mode='trilinear')
			imgs = imgs.squeeze(0)

		return imgs, label, img_paths

	def name(self):
		return "OursDataset"

	def __get_from_txt(self, dataroot):
		img_paths, labels = [], []
		with open(dataroot, 'r') as tf:
			lines = tf.readlines()
			for line in lines:
				# img_path, subid, label
				path, subid, label = line.split('\t')
				# change path to absolute path
				label = int(label)
				img_paths.append(path)
				labels.append(label)

		return img_paths, labels
		

class AbideDataset(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		# read the paths and labes from txt
		self.img_paths, self.labels = self.__get_from_txt(opt.dataroot)
		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		img_paths = self.img_paths[index]
		label = self.labels[index]
		imgs = nib.load(img_paths)
		imgs = imgs.get_data()
		imgs = torch.from_numpy(imgs)
		imgs = imgs.float().unsqueeze(0).unsqueeze(0)
		imgs = F.interpolate(imgs, size=(156,256,256), mode='trilinear')
		imgs = imgs.squeeze(0)

		return imgs, label, img_paths

	def name(self):
		return "AbideDataset"

	def __get_from_txt(self, dataroot):
		img_paths, labels = [], []
		with open(dataroot, 'r') as tf:
			lines = tf.readlines()
			for line in lines:
				# img_path, subid, label
				path, subid, label = line.split('\t')
				# change path to absolute path
				# /home/dao2/Desktop/fmri/dataset/abide
				root = '/home/dao2/Desktop/fmri/dataset/abide'
				path = path.replace('../sort_data', root)
				label = int(label)
				img_paths.append(path)
				labels.append(label)

		return img_paths, labels

class AbideDatasetTest(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths, self.labels = self.__get_from_txt(opt.dataroot)
		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.img_paths)
	
	def __getitem__(self, index):
		img_paths = self.img_paths[index]
		label = self.labels[index]
		imgs = nib.load(img_paths)
		imgs = imgs.get_data()
		imgs = torch.from_numpy(imgs)
		imgs = torch.float().unsqueeze(0).unsqueeze(0)
		imgs = F.interpolate(imgs, size=self.opt.s, mode='trilinear')
		imgs = imgs.squeeze(0)
		return imgs, label, img_paths

	def name(self):
		return 'AbideTestDataset'

	def __get_from_txt(self, dataroot):
		img_paths, labels = [], []
		with open(dataroot, 'r') as tf:
			lines = tf.readlines()
			for line in lines:
				path, subid, label = line.split('\t')
				root = '/home/dao2/Desktop/fmri/dataset/abide'
				path = path.replace('../sort_data', root)
				label = int(label)
				img_paths.append(path)
				labels.append(label)

		return img_paths, labels