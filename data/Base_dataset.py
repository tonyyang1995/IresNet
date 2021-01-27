import numpy as np
import torch
import torch.utils.data as data

class BaseDataset(data.Dataset):
	def __init__(self, opt):
		self.opt = opt

	def __len__(self):
		pass

	def __getitem__(self, index):
		pass

	def name(self):
		return "base dataset"

	def resize(self, img, size, mode='bilinear'):
		img_size = img.size()
		if (len(img_size) == 3):
			img = img.unsqueeze(0)
		img = F.interpolate(img, size=size, mode=mode).squeeze(0)
		return img

	def pad_to_square(self, img, pad_value):
		# this function makes an image into square shape
		c, h , w = img.shape
		dim_diff = np.abs(h - w)
		pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
		pad = (0,0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
		img = F.pad(img, pad, 'constant', value=pad_value)
		return img, pad