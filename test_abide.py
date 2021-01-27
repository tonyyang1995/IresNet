from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import numpy as np 

from opt.Options import Options
from data.get_dataset import get_dataset
from models.model import create_model

import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def print_loss(loss, epoch, total_epoch, cur_iter, total_iter):
	message = '\n--------------------[Epoch %d/%d, Batch %d/%d]--------------------\n' % (epoch, total_epoch, cur_iter, total_iter)
	message += '{:>10}\t{:>10.4f}\n'.format('loss:', loss)
	message += '--------------------------------------------------------------------\n'
	print(message)

def val(epoch):
	logger = open('abide_log.txt', 'w')
	opt = Options().parse()
	#opt.dataroot = 'dataset/abide/labels/test_abide_7.txt'

	opt.dataset_mode = 'abide'
	opt.name = 'abide'

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
	TP, FP, FN, total = 0, 0, 0, 0
	TN = 0
	label_tp, label_tn = 0, 0
	start_time = time.time()
	for i, (imgs, labels, img_paths) in enumerate(valLoader):
		inputs = {'img': imgs}

		name = img_paths[0].split('/')[-1]
		model.set_input(inputs, mode='test')
		output = model.inference().detach().cpu()		

		#print(output, labels, name)
		logger.write("%d\t%d\t%s\n"%(int(output), int(labels), name))
		if labels.data == 1:
			label_tp += 1
		if labels.data == 0:
			label_tn += 1

		if output.data == 1 and labels.data == 1:
			TP += 1
		elif output.data == 0 and labels.data == 0:
			TN += 1
		elif output.data == 1 and labels.data == 0:
			FP += 1
		elif output.data == 0 and labels.data == 1:
			FN += 1
	if TP+FP == 0:
		P = 0
	else:
		P = float(TP) / (TP + FP)
	
	if TP+FN == 0:
		R = 0
	else:
		R = float(TP) / (TP + FN)

	if P+R == 0:
		F1 = 0
	else:
		F1 = (2*P*R) / (P + R)

	sen = float(TP) / (TP + FN)
	spe = float(TN) / (TN + FP)
	acc = (float(TP) + float(TN))/ len(valLoader) * 100.0
	end_time = time.time()

	#print(TP + TN + FP + FN)
	#print(len(valLoader))

	message = '\n------------------------results----------------------\n'
	# message += '{:>10}\t{:>10}\t{:>10}\n'.format('TP:', TP, label_tp)
	# message += '{:>10}\t{:>10}\t{:>10}\n'.format('TN:', TN, label_tn)
	message += '{:>10}\t{:>10.4f}\n'.format('acc:', acc)
	message += '{:>10}\t{:>10.4f}\n'.format('precision:', P)
	message += '{:>10}\t{:>10.4f}\n'.format('recall:', R)
	message += '{:>10}\t{:>10.4f}\n'.format('Specificity:', spe)
	message += '{:>10}\t{:>10.4f}\n'.format('Sensitivity:', sen)
	message += '{:>10}\t{:>10.4f}\n'.format('F1-measure:', F1)
	message += '{:>10}\t{:>10.4f}\n'.format('avg_time:', (end_time - start_time) / len(valLoader))
	message += '------------------------------------------------------\n'
	print(message)
	logger.write(message + '\n')
	return acc

if __name__ == '__main__':
	val("best")
