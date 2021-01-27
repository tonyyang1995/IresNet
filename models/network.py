import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
	if isinstance(module, nn.Conv2d):
		nn.init.kaiming_normal_(module.weight.data, model='fan_out')
	elif isinstance(module, nn.BatchNorm2d):
		module.weight.data.fill_(1)
		module.bias.data.zero_()
	elif isinstance(module, nn.Linear):
		module.bias.data.zero_()

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(BasicBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.downsample = downsample
		self.stride = stride
		# self.shortcut = nn.Sequential()
		# if in_channels != out_channels:
		# 	self.shortcut.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
		# 	self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.bn2(self.conv2(out))
		#out += self.shortcut(x)
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)

		return out

class BottoleneckBlock(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(BottleneckBlock, self).__init__()

		bottleneck_channels = out_channels // self.expansion

		self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(bottleneck_channels)

		self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(bottleneck_channels)

		self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		# self.shortcut = nn.Sequential()
		# if in_channels != out_channels:
		# 	self.shortcut.add_module(
		# 		'conv',
		# 		nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
		# 	)
		# 	self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)
		return out

class resnet_2d(nn.Module):
	def __init__(self, block, layers, num_classes, last_fc=True):
		self.inplanes = 64
		self.last_fc = last_fc
		super(resnet_2d, self).__init__()
		self.conv1 = nn.Conv2d(156, 256, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, layers[0], 256, stride=1)
		self.layer2 = self._make_layer(block, layers[1], 512, stride=2)
		self.layer3 = self._make_layer(block, layers[2], 512, stride=2)
		self.layer4 = self._make_layer(block, layers[3], 1024, stride=2)

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = nn.Linear(1024 * block.expansion, num_classes)

	def _make_layer(self, block, blocks, planes, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		print(x.size())
		x = self.layer1(x)
		print(x.size())
		x = self.layer2(x)
		print(x.size())
		x = self.layer3(x)
		print(x.size())
		x = self.layer4(x)
		print(x.size())
		x = self.avgpool(x)
		x = torch.flatten(x,1)
		#print(x.size())
		x = self.fc(x)
		#print(x.size())
		#x = F.softmax(x, dim=0)
		#print(x.size())
		return x

def resnet18(opt):
	return resnet_2d(BasicBlock, [2,2,2,2], opt.num_classes)

def resnet34(opt):
	return resnet_2d(BasicBlock, [3,4,6,3], opt.num_classes)

def resnet50(opt):
	return resnet_2d(BottleneckBlock, [3,4,6,3], opt.num_classes)

def resnet101(opt):
	return resnet_2d(BottleneckBlock, [3,4,23,3], opt.num_classes)

def resnet152(opt):
	return resnet_2d(BottleneckBlock, [3,8,36,3], opt.num_classes)
