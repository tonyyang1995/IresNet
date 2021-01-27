import os
import time
import torch
import numpy as np 
import statistics 
from opt.Options import Options
from data.get_dataset import get_dataset
from models.model import create_model

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F

from scipy import stats
import numpy as np 
import logging, sys
import csv
import logging
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def val(epoch, np_save_name, dataroot, dataset_mode, opt_name):
	opt = Options().parse()
	opt.dataroot = dataroot

	opt.dataset_mode = dataset_mode

	opt.name = opt_name

	opt.load_model = os.path.join(opt.checkpoint_dir, opt.name, opt.name+'_'+str(epoch)+'.pth')
	print(opt.load_model)

	dataset = get_dataset(opt)
	valLoader = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		shuffle=False
	)
	print(len(valLoader))
	model = create_model(opt)
	if len(opt.gpu_ids) > 1:
		model = model.module

	model.load(opt.load_model)

	start_time = time.time()

	dicts = {}
	for i, (imgs, labels, img_paths) in enumerate(valLoader):
		st = time.time()
		inputs = {'img': imgs}
		if opt.dataset_mode == 'abide':		    # abide
			name = img_paths[0].split('/')[-2]
		elif opt.dataset_mode == 'Ours': 		# ours
			name = img_paths[0].split('/')[-1][:-7]
		# print(name)

		model.set_input(inputs, mode='test')
		output = model.inference().cpu()

		if output.data == 1:
			dicts[name] = 'yes'
		else:
			dicts[name] = 'no'

		conv1, conv2, conv5 = model.show_middle_results()
		conv5 = conv5.cpu().detach()
		input_d, input_h, input_w = 182,218,182

		bs, channels, d, h, w = conv5.size()
		cam5 = torch.zeros((1,1,d,h,w), dtype=torch.float32)

		weights = model.get_weights('module.fc.weight')
		if output.data == 1:
			weights = weights[1, ...]
		else:
			weights = weights[0, ...]

		for c in range(channels):
			cam5[0,0,...] += conv5[0, c, ...] #* weights[c]

		res5 = (cam5 - torch.min(cam5)) / (torch.max(cam5) - torch.min(cam5))
		res5 = F.interpolate(res5, size=(input_d, input_h, input_w), mode='trilinear')
		res5 = res5.squeeze(0); res5 = res5.squeeze(0)
		
		# standardize
		res5_np = res5.numpy()

		if not os.path.exists(os.path.join(np_save_name)):
			os.makedirs(os.path.join(np_save_name))
		np.save(os.path.join(np_save_name, name+'.npy'), res5_np)
	return dicts

def get_abide_paths(root, labels):
	asdn, asdp = [], []
	lines = open(labels, 'r').readlines()
	dicts = {}
	for line in lines:
		conts = line.strip().split('\t')
		path = conts[0]; gt = int(conts[2])
		name = path.split('/')[4]
		#print(name, gt)
		dicts[name] = gt
	for r, dirs, names in os.walk(root):
		for name in names:
			if dicts[name[:-4]] == 1:
				asdp.append(os.path.join(root, name))
			else:
				asdn.append(os.path.join(root, name))
	return asdn, asdp

if __name__ == '__main__':
	# this is for np_abide_test
	np_save_name = 'np_abide_test'
	dicts = val("best", np_save_name, 'dataset/abide/labels/test7.txt', 'abide', 'abide')

	# this is for np_abide_train
	# np_save_name = 'np_abide_train'
	# dicts = val("best", np_save_name, 'dataset/abide/labels/train7.txt', 'abide', 'abide')

	asdn_npy, asdp_npy = get_abide_paths(np_save_name, 'dataset/abide/labels/test7.txt')
	# this is to load abide_train_path
	#asdn_npy, asdp_npy = get_abide_paths(np_save_name, 'dataset/abide/labels/train7.txt')

	asdn_npy = sorted(asdn_npy)
	asdp_npy = sorted(asdp_npy)

	print(len(asdn_npy), len(asdp_npy))