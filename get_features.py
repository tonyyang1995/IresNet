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


def draw_train_test_val(train_t, val_t, test_t):
	train = open(train_t, 'r')
	val = open(val_t, 'r')
	test = open(test_t, 'r')

	train_lines = train.readlines()
	val_lines = val.readlines()
	test_lines =test.readlines()

	#for tl, vl, testl in zip(train_lines, val_lines, test_lines):
	x = []
	y1, y2, y3 = [], [], []
	y4, y5, y6 = [], [], []
	for i in range(len(train_lines)):
		train_acc, train_loss = train_lines[i].strip().split(',')
		val_acc, val_loss = val_lines[i].strip().split(',')
		test_acc, test_loss = test_lines[i].strip().split(',')
		x.append(i+1)
		y1.append(float(train_acc)); y2.append(float(val_acc)); y3.append(float(test_acc))
		y4.append(float(train_loss)); y5.append(float(val_loss)); y6.append(float(test_loss))

	fig, ax = plt.subplots()
	loss1 = ax.plot(x, y4, label='train_loss', markersize=2)
	ax.legend(fontsize=10, loc='center right', bbox_to_anchor=(1.1, 0.5))
	fig.savefig("train_loss.png")

	fig2, ax2 = plt.subplots()
	acc2 = ax2.plot(x, y2, label='val_acc', markersize=2)
	ax2.legend(fontsize=10, loc='center right', bbox_to_anchor=(1.1, 0.5))
	fig2.savefig("val_acc.png")

	fig3, ax3 = plt.subplots()
	acc3 = ax3.plot(x, y3, label='test_acc', markersize=2)
	ax3.legend(fontsize=10, loc='center right', bbox_to_anchor=(1.1, 0.5))
	fig3.savefig("test_acc.png")

def get_model_res(dataLoader, model, logger):
	st = time.time()
	acc = 0
	avg_loss = []
	TP, FP, FN, total = 0, 0, 0, 0
	count = 0
	for i, (imgs, labels, img_paths) in enumerate(dataLoader):
		inputs = {'img': imgs, 'labels': labels}
		count += imgs.size(0)
		#name = img_paths[0].split('/')[-2]
		model.set_input(inputs, mode='draw')
		output = model.inference_batch()
		loss = model.cal_current_loss()
		avg_loss.append(float(loss))
		for i in range(len(output)):
			if output[i].data.cpu() == labels[i].data:
				TP += 1

	acc = float(TP) / (count) * 100.0
	avg_l = sum(avg_loss) / len(avg_loss)
	print(acc, avg_l)
	logger.write('%.4f, %.4f\n'%(acc, avg_l))
	et = time.time()
	print(et - st)

def save_mid_report(start, end, train_t, val_t, test_t):
	# write to txt file
	val_txt = open(val_t, 'w')
	test_txt = open(test_t, 'w')


	train_opt = Options().parse()
	val_opt = Options().parse()
	test_opt = Options().parse()

	# dataroot
	val_opt.dataroot = 'dataset/abide/labels/test7.txt'
	# test_opt.dataroot = 'dataset/abide/labels/ours_all.txt'

	train_opt.dataroot = 'dataset/abide/labels/ours_train.txt'
	#val_opt.dataroot = 'dataset/abide/labels/ours_test.txt'
	test_opt.dataroot = 'dataset/abide/labels/less9.txt'

	# dataset mode
	# train_opt.dataset_mode = 'abide'
	# val_opt.dataset_mode = 'abide'
	# test_opt.dataset_mode = 'Ours'

	train_opt.dataset_mode = 'Ours'
	val_opt.dataset_mode = 'abide'
	test_opt.dataset_mode = 'abide'

	# model name
	#train_opt.name = 'abide_all'
	train_opt.name = 'ours_split'

	# load models
	for epoch in range(start, end):
		train_opt.load_model = os.path.join(train_opt.checkpoint_dir, train_opt.name, train_opt.name+'_'+str(epoch)+'.pth')
		test_opt.load_model = train_opt.load_model
		val_opt.load_model = train_opt.load_model
		print(train_opt.load_model)

		# train_dataset = get_dataset(train_opt)
		val_dataset = get_dataset(val_opt)
		test_dataset = get_dataset(test_opt)

		# trainLoader = torch.utils.data.DataLoader(
		# 	train_dataset,
		# 	batch_size=8,
		# 	shuffle=False
		# )
	
		valLoader = torch.utils.data.DataLoader(
			val_dataset,
			batch_size=4,
			shuffle=False
		)

		testLoader = torch.utils.data.DataLoader(
			test_dataset,
			batch_size=4,
			shuffle=False
		)
	
		#print(len(train_dataset), len(val_dataset), len(test_dataset))
	
		#train_model = create_model(train_opt)
		model = create_model(test_opt)

		if len(train_opt.gpu_ids) > 1:
			model = model.module

		model.load(test_opt.load_model)
		#get_model_res(trainLoader, model, train_txt)
		get_model_res(valLoader, model, val_txt)
		get_model_res(testLoader, model, test_txt)
	
	#train_txt.close()
	val_txt.close()
	test_txt.close()


def save_csv_mean(p, dicts, csv_name='features.csv', group="ASD", type='Training', add=False):
	#path = os.path.join(p, os.listdir(p)[0])
	path = p
	name = p.split('/')[-1][:-4]
	features = np.load(path)
	features = torch.from_numpy(features)
	# coronal
	coronal = features.clone().permute(2,1,0)
	sigittal = features.clone()
	axial = features.clone().permute(1,2,0)
	c_d = coronal.size(0)
	s_d = sigittal.size(0)
	a_d = axial.size(0)
	# print(c_d, s_d, a_d)

	if add == False:
		# create header first then write the first row
		with open(csv_name, 'w') as cf:
			f_csv = csv.writer(cf)
			header = ['names', 'Group', "Type", "AI_Outputs"]
			rows = [name, group, type, dicts[name]]
			for i in range(c_d):
				header.append('coronal_%d' %(i))
				plane_m = torch.mean(coronal[i, ...])
				rows.append(float(plane_m))
			for i in range(s_d):
				header.append('sagittal_%d' %(i))
				plane_m = torch.mean(sigittal[i, ...])
				rows.append(float(plane_m))
			for i in range(a_d):
				header.append('axial_%d' %(i))
				plane_m = torch.mean(axial[i, ...])
				rows.append(float(plane_m))
			f_csv.writerow(header)
			f_csv.writerow(rows)
	else:
		with open(csv_name, 'a') as cf:
			f_csv = csv.writer(cf)
			rows = [name, group, type, dicts[name]]
			for i in range(c_d):
				plane_m = torch.mean(coronal[i, ...])
				rows.append(float(plane_m))
			for i in range(s_d):
				plane_m = torch.mean(sigittal[i, ...])
				rows.append(float(plane_m))
			for i in range(a_d):
				plane_m = torch.mean(axial[i, ...])
				rows.append(float(plane_m))
			f_csv.writerow(rows)

def group_test(m1, m2, threshold, name, ax): # m1 = n * slices
	#print(name)
	xs = []; 
	y1_mean = []; y1_std = []
	y2_mean = []; y2_std = []
	ps = []
	for i in range(len(m2[0])):
		asd = []
		td = []
		for j in range(len(m2)):
			asd.append(m2[j][i])
		for j in range(len(m1)):
			td.append(m1[j][i])
		asd_mean = statistics.mean(asd)
		asd_std = statistics.stdev(asd)
		td_mean = statistics.mean(td)
		td_std = statistics.stdev(td)
		#if asd_mean < threshold:
		#	continue
		#else:
		xs.append(i)
		y1_mean.append(td_mean); y1_std.append(td_std)
		y2_mean.append(asd_mean); y2_std.append(asd_std)
		p = ttest(asd, td, "asdp vs asdn", threshold, flag=False)
		#print(i, p)
		ps.append(p)

	width = 0.4
	labels = []
	asdn_mean = []; asdn_std = []
	asdp_mean = []; asdp_std = []
	all_p = []

	#asdn_means = []; asdp_means = []
	#xs = np.array(xs)
	#rec1 = ax.bar(xs-width/2, y1_mean, width, label='ASDN', yerr=y1_std)
	#rec2 = ax.bar(xs+width/2, y2_mean, width, label='ASDP', yerr=y2_std)

	for i in range(len(xs)):
		if y2_mean[i] > threshold and ps[i] < 0.05:
			labels += [str(i)]
			asdn_mean.append(y1_mean[i]); asdn_std.append(y1_std[i])
			asdp_mean.append(y2_mean[i]); asdp_std.append(y2_std[i])
			all_p.append(ps[i])
			#ax.annotate('*', xy=(xs[i], max(y1_mean[i], y2_mean[i])+0.3), xytext=(xs))
		elif y1_mean[i] > threshold and ps[i] < 0.05:
			labels += [str(i)]
			asdn_mean.append(y1_mean[i]); asdn_std.append(y1_std[i])
			asdp_mean.append(y2_mean[i]); asdp_std.append(y2_std[i])
			all_p.append(ps[i])

	xs = np.arange(len(labels))
	ax.set_ylabel(name)
	ax.set_xticks(xs)
	ax.set_xticklabels(labels)
	rec1 = ax.bar(xs-width/2, asdn_mean, width, label='ASDN', yerr=asdn_std)
	rec2 = ax.bar(xs+width/2, asdp_mean, width, label='ASDP', yerr=asdp_std)

	for i in range(len(all_p)):
		ax.annotate('*', xy=(xs[i]-width/2, max(asdn_mean[i], asdp_mean[i])+0.01))
	ax.legend(fontsize=10, loc="center right", bbox_to_anchor=(1.1,0.5))


def get_means(paths):
	c_means = []
	s_means = []
	a_means = []
	for path in paths:
		mean = []
		#p = os.path.join(path, os.listdir(path)[0])
		p = path
		features = np.load(p)
		features = torch.from_numpy(features)
		coronal = features.clone().permute(2,1,0)
		sagittal = features.clone()
		axial = features.clone().permute(1,2,0)
		# if pos == 'coronal':
		# 	features = features.clone().permute(2,1,0)
		# elif pos == 'sagittal':
		# 	features = features.clone()
		# else:
		# 	features = features.clone().permute(1,2,0)
		d, h, w  = coronal.size()
		for i in range(d):
			m  = torch.mean(coronal[i, ...])
			mean.append(float(m))
		c_means.append(mean)

		d, h, w = sagittal.size()
		mean = []
		for i in range(d):
			m = torch.mean(sagittal[i, ...])
			mean.append(float(m))
		s_means.append(mean)

		d, h, w = axial.size()
		mean = []
		for i in range(d):
			m = torch.mean(axial[i, ...])
			mean.append(float(m))
		a_means.append(mean)

	return c_means, s_means, a_means

def get_mean(p, Type="mean", pos="coronal", threshold=0.0):
	means = []
	idxs = {}

	overall = None
	for i in range(len(p)):
		#path = os.path.join(p[i], os.listdir(p[i])[0])
		#path = p[i]
		#print(p[i])
		name = p[i].split('/')[-1][-7]
		features = np.load(p[i])
		features = torch.from_numpy(features)
		if pos == 'coronal':
			features = features.permute(2,1,0)
		elif pos == 'sigittal':
			features = features
		elif pos == 'axial':
			features = features.permute(1,2,0)

		overall = features if overall is None else overall + features
		channel, h, w = features.size()
		m = 0
		count = 0
		idx = []
		for c in range(channel):
			plane = features[c, ...]
			plane_m = torch.mean(plane)
			#print(plane_m)
			#plane_m = torch.max(plane)
			if plane_m < threshold:
				continue
			#m = m + float(torch.mean(features[c, ...])) if Type == 'mean' else m + float(torch.max(features[c, ...]))
			m += plane_m
			idx.append(c)
			count += 1
			#means.append(float(plane_m))
		if count:
			m /= count
			means.append(m)
			idxs[name] = idx

	overall = overall / len(p)
	overall_m = 0
	overall_count = 0
	for c in range(overall.size(0)):
		plane = overall[c, ...]
		plane_m = torch.mean(plane)
		if plane_m < threshold:
			continue
		overall_m += plane_m
		overall_count += 1
	if overall_count:
		overall_m /= overall_count

	return means, idxs, overall_m

def ttest(v1, v2, name, threshold, flag=True):
	if len(v1) == 0 or len(v2) == 0:
		print("the vectors are empty!")
	t, p = stats.ttest_ind(v1, v2)
	message = '\n----------------T-Test Results of %s----------------------\n' %(name)
	message += '{:>10}\t{:>10.4f}\n'.format('Threshold: ', threshold)
	message += '{:>10}\t{:>10.4f}\n'.format('T-value:', t)
	message += '{:>10}\t{:>10.4f}\n'.format('P-value: ', p)
	message += '-------------------------------------------------------------\n'
	if flag:
		print(message)
	return p

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

def get_paths(root):
	asdn, asdp = [], []
	for r, dirs, names in os.walk(root):
		for name in names:
			#if not 'nii.gz' in name:
			#	continue
			if '.DS_Store' in name:
				continue
			if 'ASDN' in name:
				asdn.append(os.path.join(root, name))
			else:
				asdp.append(os.path.join(root, name))
	return asdn, asdp

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

def calculate_means(paths):
	feat = None
	for path in paths:
		#p = os.path.join(path, os.listdir(path)[0])
		if '.DS_Store' in path:
			continue
		features = np.load(path)
		features = torch.from_numpy(features)

		feat = features if feat is None else feat + features

	feat = feat / len(paths)
	
	cor = feat.clone().permute(2,1,0)
	sig = feat.clone()
	axi = feat.clone().permute(1,2,0)
	return cor, sig, axi

def add_threshold(m1, m2, thresh=0.2):
	# return xs, ys, threshs
	xs = []
	y1 = []
	y2 = []
	thr = []
	d, h, w = m1.size()
	for i in range(d):
		mean1 = torch.mean(m1[i, ...])
		mean2 = torch.mean(m2[i, ...])
		xs.append(i)
		y1.append(float(mean1))
		y2.append(float(mean2))
		thr.append(thresh)
	return xs, y1, y2, thr


def draw_mean(p1, p2, save_name="mean.pdf"):
	# load the features and calculate the mean
	n_cor, n_sig, n_axi = calculate_means(p1)
	p_cor, p_sig, p_axi = calculate_means(p2)

	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(1,1,1)

	threshold = 0.0
	x, y1, y2, thr = add_threshold(n_cor, p_cor, thresh=threshold)
	
	f = open('roc_data/p_value_coronal.txt', 'w')
	for i in range(50, 230):
		mess = '%d\t%.4f\t%.4f\n' %(x[i], y1[i], y2[i])
		f.write(mess)
	f.close()

	ax.plot(x[50:230], y1[50:230], "*", label="TD in Coronal MRI", markersize=2, c='b')
	ax.plot(x[50:230], y2[50:230], "*", label='ASD in Coronal MRI', markersize=2, c='r')

	ax.set_xlabel('Different Slices on Coronal Area', fontsize=10)
	ax.set_ylabel('mean of RNLY Model Value Per Slice', fontsize=10)
	#ax.set_title('2D visual of activation number on coronal, the plane size is %d * %d. The threshold is %.2f' %(n_cor.size(1), n_cor.size(2), threshold), fontsize=30)
	#ax.legend(fontsize=10, loc='center right', bbox_to_anchor=(1.1, 0.5))
	ax.legend(fontsize=10)
	plt.savefig('coronal.png')

	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(1,1,1)
	threshold = 0.0
	x, y1, y2, thr = add_threshold(n_sig, p_sig, thresh=threshold)
	
	f = open('roc_data/p_value_sagittal.txt', 'w')
	for i in range(20, 150):
		mess = '%d\t%.4f\t%.4f\n' %(x[i], y1[i], y2[i])
		f.write(mess)
	f.close()

	ax.plot(x[20:150], y1[20:150], "*", label='TD in Sagittal MRI', markersize=2, c='b')
	ax.plot(x[20:150], y2[20:150], "*", label='ASD in Sagittal MRI', markersize=2, c='r')
	#ax.plot(x[20:150], thr[20:150], label='threshold', c = 'k')
	#ax.tick_params(axis='x', labelsize=20)
	#ax.tick_params(axis='y', labelsize=20)
	ax.set_xlabel('Different Slices on Sagittal Area', fontsize=10)
	ax.set_ylabel('mean of RNLY Model Value Per Slice', fontsize=10)
	#ax.set_title('2D visual of activation number on coronal, the plane size is %d * %d. The threshold is %.2f' %(n_sig.size(1), n_sig.size(2), threshold), fontsize=30)
	#ax.legend(fontsize=20, loc='center right', bbox_to_anchor=(1.1, 0.5))
	ax.legend(fontsize=10)
	plt.savefig('sagittal.png')

	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(1,1,1)
	threshold = 0.0
	x, y1, y2, thr = add_threshold(n_axi, p_axi, thresh=threshold)
	
	f = open('roc_data/p_value_axial.txt', 'w')
	for i in range(20, 160):
		mess = '%d\t%.4f\t%.4f\n' %(x[i], y1[i], y2[i])
		f.write(mess)
	f.close()

	ax.plot(x[20:160], y1[20:160], "*", label='TD in Axial MRI', markersize=2, c='b')
	ax.plot(x[20:160], y2[20:160], "*", label='ASD in Axial MRI', markersize=2, c='r')
	#ax.plot(x[20:160], thr[20:160], label='threshold', c='k')
	#ax.tick_params(axis='x', labelsize=20)
	#ax.tick_params(axis='y', labelsize=20)
	ax.set_xlabel('Different Slices on Axial Area', fontsize=10)
	ax.set_ylabel('mean of RNLY Model Value Per Slice', fontsize=10)
	#ax.set_title('2D visual of activation number on coronal, the plane size is %d * %d. The threshold is %.2f' %(n_axi.size(1), n_axi.size(2), threshold), fontsize=30)
	#ax.legend(fontsize=20, loc='center right', bbox_to_anchor=(1.1, 0.5))
	ax.legend(fontsize=10)
	plt.savefig("axial.png")
	#plt.show()

def remove_threshold(m1, m2, thresh=0.2):
	y1 = []
	y2 = []
	idxs = []
	d, h ,w = m1.size()
	for i in range(d):
		# if torch.max(m1[i, ...] < 0.7) or torch.max(m2[i, ...] < 0.7):
		# 	print(i)
		# 	continue
		mean1 = torch.mean(m1[i, ...])
		mean2 = torch.mean(m2[i, ...])		
		if mean2 > thresh:
			y1.append(float(mean1))
			y2.append(float(mean2))
			idxs.append(i)
		#if mean1 > thresh:
			# y1.append(float(mean1))

	return y1, y2, idxs

def groups_pvalue(p1, p2):
	n_cor, n_sig, n_axi = calculate_means(p1)
	p_cor, p_sig, p_axi = calculate_means(p2)

	threshold=0.0
	y1, y2, _ = remove_threshold(n_cor, p_cor, thresh=threshold)
	ttest(y1, y2, "asdn vs asdp coronal", threshold)

	threshold=0.0
	y1, y2, _ = remove_threshold(n_sig, p_sig, thresh=threshold)
	ttest(y1, y2, "asdn vs asdp sagittal", threshold)

	threshold=0.0
	y1, y2, _ = remove_threshold(n_axi, p_axi, thresh=threshold)
	ttest(y1, y2, "asdn vs asdp axial", threshold)

def onettest(v1, mean, name, threshold, flag=True):
	if len(v1) == 0:
		print('the vector is empty')
	t, p = stats.ttest_1samp(v1, mean)
	message = "\n ----One sample T-Test among the groups of %s -------------------\n" %(name)
	message += '{:>10}\t{:>10.4f}\n'.format('mean: ', mean)
	#message += '\n----------------T-Test Results of %s----------------------\n' %(name)
	message += '{:>10}\t{:>10.4f}\n'.format('Threshold: ', threshold)
	message += '{:>10}\t{:>10.4f}\n'.format('T-value:', t)
	message += '{:>10}\t{:>10.4f}\n'.format('P-value: ', p)
	message += '---------------------------------------------------------------------\n'
	if flag:
		print(message)
	return p

def chisquare(v1, name, flag=True):
	from scipy.stats import chi2_contingency
	chi2, p, dof, ex = chi2_contingency(v1)
	message = "\n ----Chisquare test among the groups of %s -------------------\n" %(name)
	message += '{:>10}\t{:>10.4f}\n'.format('chi2:', chi2)
	message += '{:>10}\t{:>10.4f}\n'.format('P-value: ', p)
	message += '---------------------------------------------------------------------\n'
	if flag:
		print(message)	
	return p

def Agroup_pvalue(p1, p2):
	#n_cor_means, _, mean1 = get_mean(p1, pos='coronal')
	#p_cor_means, _, mean2 = get_mean(p2, pos='coronal')
	#onettest(n_cor_means, mean1, "among asdn in coronal", threshold)
	#onettest(p_cor_means, mean2, "among asdp in coronal", threshold)
	n_cor, n_sig, n_axi = get_means(p1)
	p_cor, p_sig, p_axi = get_means(p2)
	
	chisquare(n_cor, "asdn coronal")
	chisquare(p_cor, "asdp coronal")

	#n_sig_means, _, mean1 = get_mean(p1, pos='sagittal')
	#p_sig_means, _, mean2 = get_mean(p2, pos='sagittal')
	#onettest(n_sig_means, mean1, 'among asdn in sagittal', threshold)
	#onettest(p_sig_means, mean2, 'among asdp in sagittal', threshold)
	chisquare(n_sig, "asdn sagittal")
	chisquare(p_sig, "asdp sagittal")
	
	#n_axi_means, _, mean1 = get_mean(p1, pos='axial')
	#p_axi_means, _, mean2 = get_mean(p2, pos='axial')
	#onettest(n_axi_means, mean1, 'among asdn in axial', threshold)
	#onettest(p_axi_means, mean2, 'among asdp in axial', threshold)
	chisquare(n_axi, "asdn axial")
	chisquare(p_axi, "asdp axial")

def draw_pvalues(p1, p2):
	n_cor, n_sig, n_axi = get_means(p1)
	p_cor, p_sig, p_axi = get_means(p2)

	fig = plt.figure(figsize=(12,2))
	#spec2 = gridspec.GridSpec(ncols=5, nrows=2,figure=fig)
	ax = fig.add_subplot(1,1,1)
	#print(len(n_cor), len(n_cor[0]))
	#print(len(p_cor), len(p_cor[0]))
	group_test(n_cor, p_cor, 0.0, "coronal", ax)
	plt.savefig("p_coronal.png")
	#ax.set_xlabel("Different Slice on Coronal Area", fontsize=10)
	#ax.set_ylabel("Mean")

	fig = plt.figure(figsize=(12,2))
	ax1 = fig.add_subplot(1,1,1)
	group_test(n_sig, p_sig, 0.0, "sagittal", ax1)
	plt.savefig("p_sagittal.png")

	fig = plt.figure(figsize=(12,2))
	ax2 = fig.add_subplot(1,1,1)
	group_test(n_axi, p_axi, 0.0, "axial", ax2)
	plt.savefig("p_axial.png")


if __name__ == '__main__':
	# s1 get the np features
	# np_save_name = 'np_abide_test_final_std/'
	# np_save_name = 'np_abide_test_weight_final/'
	# np_save_name = 'np_ours_split_train_final'
	
	#  save ours test
	np_save_name = 'np_ours_test'
	dicts = val(115, np_save_name, 'dataset/abide/labels/ours_test_reori.txt', 'Ours', 'ours_split_reori')
	
	# save abide test
	#np_save_name = 'np_abide_test'
	#dicts = val(59, np_save_name, 'dataset/abide/labels/test7.txt', 'abide', 'abide_all')

	
	# s2 draw corinal, sagittal and axial
	asdn_npy, asdp_npy = get_paths(np_save_name)

	print(len(asdn_npy), len(asdp_npy))
	asdn_npy = sorted(asdn_npy)
	asdp_npy = sorted(asdp_npy)

	# s3 save csv
	csv_name = 'ours_test_reori_features.csv'


	# t = "training"
	t = 'testing'
	# print(len(asdn_npy) + len(asdp_npy))
	flag = True
	if len(asdn_npy) == 0:
		flag = False
	for i in range(len(asdn_npy)):
		if i == 0:
			save_csv_mean(asdn_npy[i], dicts, csv_name=csv_name, group="TD", type=t, add=False)
		else:
			save_csv_mean(asdn_npy[i], dicts, csv_name=csv_name, group="TD", type=t, add=True)
	for i in range(len(asdp_npy)):
		if flag == False:
			save_csv_mean(asdp_npy[i], dicts, csv_name=csv_name, group="ASD", type=t, add=False)
			flag = True
		else:
			save_csv_mean(asdp_npy[i], dicts, csv_name=csv_name, group="ASD", type=t, add=True)
