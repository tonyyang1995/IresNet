import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import math
from functools import partial

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(BasicBlock, self).__init__()

		self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm3d(out_channels)

		self.downsample = downsample
		self.stride = stride

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
		super(BottoleneckBlock, self).__init__()

		bottleneck_channels = out_channels // self.expansion

		self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm3d(bottleneck_channels)

		self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm3d(bottleneck_channels)

		self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm3d(out_channels)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

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

class resnet_3d(nn.Module):
	def __init__(self, block, layers, num_classes, last_fc=True):
		self.inplanes = 64
		self.last_fc = last_fc
		super(resnet_3d, self).__init__()
		# self.conv1 = nn.Conv3d(self.opt.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
		self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
		self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
		self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
		self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

		self.fc = nn.Linear(512 * block.expansion, num_classes)
		# self.fc2 = nn.Linear(512 * block.expansion, 100)
		# self.fc3 = nn.Linear(100, num_classes)

	def _make_layer(self, block, blocks, planes, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm3d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		#print('conv1')
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		conv1 = x

		#print('layers1')
		x = self.layer1(x)
		conv2 = x
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		conv5 = x
		x = self.avgpool(x)
		x = torch.flatten(x,1)
		x = self.fc(x)
		return x, conv1, conv2, conv5