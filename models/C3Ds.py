import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .C3D import C3D, t2CC3D, vgg_3D
from .BaseModel import BaseModel

from torch.autograd import Variable

class ThreeDNet(BaseModel):
	def name(self):
		return 'c3d'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.cirterion = nn.CrossEntropyLoss()
		#self.cirterion = nn.BCELoss()

		self.opt = opt
		if opt.model_name == 'c3d':
			self.model = C3D(num_classes=opt.num_classes)
		elif opt.model_name == '2cc3d':
			self.model = t2CC3D(num_classes=opt.num_classes)
		elif opt.model_name == 'vgg_3D':
			self.model = vgg_3D(num_classes=opt.num_classes)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
		if len(self.opt.gpu_ids) > 1:
			self.model = torch.nn.DataParallel(self.model).to(opt.device)
		elif len(self.opt.gpu_ids) > 0:
			self.model = self.model.to(opt.device)


	def set_input(self, input, mode='train'):
		self.imgs = Variable(input['img'].permute(0,2,1,3,4).to(self.opt.device))
		if mode == 'train':
			self.labels = Variable(input['labels'].to(self.opt.device))

	def get_current_loss(self):
		return self.loss, self.outputs

	def forward(self):
		self.outputs = self.model(self.imgs)

	def backward(self):
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

	def inference(self):
		self.model.eval()
		output = self.model(self.imgs)
		return output.argmax()

	def optimize_parameters(self):
		self.forward()
		self.loss = self.cirterion(self.outputs, self.labels)
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