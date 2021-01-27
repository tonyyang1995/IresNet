import torch
from torch import nn

from . import resnet

def create_model(opt):
	if opt.model_name == 'resnet2d':
		from .ResNet2D import resnet2d
		model = resnet2d()
	elif opt.model_name == 'alexnet2d':
		from .AlexNet import alexnet2d
		model = alexnet2d()
	elif opt.model_name == 'c3d' or opt.model_name == '2cc3d' or opt.model_name == 'vgg_3D':
		from .C3Ds import ThreeDNet
		model = ThreeDNet()
	elif opt.model_name == 'resnet3d':
		from .ResNet3D import resnet3d
		model = resnet3d()
	else:
		raise NotImplementedError("model depth should be in [10, 18, 34, 50, 101, 152, 200], but got %d" % (opt.model_depth))

	model.initialize(opt)
	print(opt.gpu_ids)
	if len(opt.gpu_ids) > 0:
		model = model.cuda()
	if len(opt.gpu_ids) > 1:
		model = nn.DataParallel(model)
	return model