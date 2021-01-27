import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import resnet_2d, BasicBlock, BottoleneckBlock
from .BaseModel import BaseModel

from torch.autograd import Variable

class AlexNet2D(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel=3, out_channel=96, kernel_size=11, stride=4)
        #self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        #self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stide=2)

        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        #self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        #self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.fc1 = nn.Linear(in_feature=256*6*6, out_feature=4096)
        self.fc2 = nn.Linear(in_feature=4096, out_feature=4096)
        self.fc3 = nn.Linear(in_feature=4096, out_featrue=num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        norm1 = self.norm1(conv1)
        pool1 = self.pool1(norm1)

        conv2 = self.relu(self.conv2(pool1))
        norm2 = self.norm2(conv2)
        pool2 = self.pool2(norm2)

        conv3 = self.relu(self.conv3(pool2))
        conv4 = self.relu(self.conv4(conv3))
        conv5 = self.relu(self.conv5(conv4))

        flat = torch.flatten(conv5,1)
        fc1 = self.relu(self.fc1(self.dropout(flat)))
        fc2 = self.relu(self.fc2(self.dropout(fc1)))
        fc3 = self.fc3(fc2)

        return fc3


        