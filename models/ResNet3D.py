import os, sys
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from .resnet import resnet_3d, BasicBlock, BottoleneckBlock
from .BaseModel import BaseModel

from torch.autograd import Variable

class resnet3d(BaseModel):
	def name(self):
		return 'resnet3d'

	def initialize(self, opt):
		#print(self.name)
		BaseModel.initialize(self, opt)
		self.criterion = nn.CrossEntropyLoss().cuda()

		self.opt = opt
		if opt.model_depth == 18:
			print('using resnet3d 18')
			self.model = resnet_3d(BasicBlock, [1,1,1,1], num_classes=opt.num_classes)
		elif opt.model_depth == 34:
			self.model = resnet_3d(BasicBlock, [3,4,6,3], num_classes=opt.num_classes)
		elif opt.model_depth == 50:
			self.model = resnet_3d(BottoleneckBlock, [3,4,6,3], num_classes=opt.num_classes)
		elif opt.model_depth == 101:
			self.model = resnet_3d(BottoleneckBlock, [3,4,23,3], num_classes=opt.num_classes)
		elif opt.model_depth == 152:
			self.model = resnet_3d(BottoleneckBlock, [3,8,36,3], num_classes=opt.num_classes)

		#self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=0.9)
		# self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr)

		if len(self.opt.gpu_ids) > 1:
			self.model = torch.nn.DataParallel(self.model).to(opt.device)
		elif len(self.opt.gpu_ids) > 0:
			self.model = self.model.to(opt.device)

	def set_input(self, input, mode='train'):
		#print(input['img'].size())
		if self.opt.perm:
			self.imgs = Variable(input['img'].permute(0,2,1,3,4).to(self.opt.device))
		else:
			self.imgs = Variable(input['img']).to(self.opt.device)

		if mode == 'train' or mode == 'draw':
			self.labels = Variable(input['labels'].to(self.opt.device))

		#print(self.imgs.size())

	def cal_current_loss(self):
		self.loss = self.criterion(self.outputs, self.labels)
		return self.loss

	def get_current_loss(self):
		return self.loss, self.outputs

	def forward(self):
		self.outputs, self.conv1, self.conv2, self.conv5 = self.model(self.imgs)
		#print(self.outputs.size())
		#print(self.labels.size())

	def show_middle_results(self):
		return self.conv1, self.conv2, self.conv5
		#return self.unflat, self.flat

	def backward(self):
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

	def inference_batch(self):
		self.model.eval()
		self.outputs, self.conv1, self.conv2, self.conv5 = self.model(self.imgs)
		output = []
		for i in range(self.outputs.size(0)):
			output.append(self.outputs[i].argmax())
		return output

	def roc(self):
		self.model.eval()
		output, self.conv1, self.conv2, self.conv5 = self.model(self.imgs)
		return output.sigmoid()

	def inference(self):
		self.model.eval()
		output, self.conv1, self.conv2, self.conv5 = self.model(self.imgs)
		return output.argmax()
		#return output.argmax(), output
		# print(self.outputs.size())
		# output = []
		# for i in range(self.outputs.size(0)):
		# 	output.append(self.outputs[i].argmax())
		# return output

	def optimize_parameters(self):
		self.forward()
		#self.loss = self.criterion(self.outputs, self.labels.squeeze(1))
		self.loss = self.criterion(self.outputs, self.labels)
		self.backward()

	def save(self, name, epoch):
		# check whether the file exists
		dirs = os.path.join(self.opt.checkpoint_dir, name)
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		path = os.path.join(self.opt.checkpoint_dir, name, name+'_'+str(epoch)+'.pth')
		torch.save(self.model.state_dict(), path)

	def load(self, model_path):
		if len(self.opt.gpu_ids) > 1:
			state_dict = torch.load(model_path)
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				if 'module' in k:
					new_state_dict[k] = v
				else:
					name = 'module.' + k
					new_state_dict[name] = v
			self.model.load_state_dict(new_state_dict)
		else:
			state_dict = torch.load(model_path)
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				if 'module.' in k:
					name = k[7:]
					new_state_dict[name] = v
				else:
					new_state_dict[k] = v
			self.model.load_state_dict(new_state_dict)

	def get_weights(self, key):
		#for k in self.model.state_dict():
		#	print(k)
		weight = self.model.state_dict()[key]
		return weight