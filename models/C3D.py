import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class C3D(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D, self).__init__()
        # 3* 128 * 128
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # 64 * 64 * 64

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 32 * 32

        self.conv3a = nn.Conv3d(64,128, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3b = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 256 * 16 * 16

        self.conv4a = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 512 * 8 * 8

        self.conv5a = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,1,1)) # 512 * 4 * 4

        self.fc6 = nn.Linear(256 * 9 * 5 * 5, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        #print(x.size())
        conv1 = self.relu(self.conv1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        conv3a = self.relu(self.conv3a(pool2))
        conv3b = self.relu(self.conv3b(conv3a))
        pool3 = self.pool3(conv3b)

        conv4a = self.relu(self.conv4a(pool3))
        conv4b = self.relu(self.conv4b(conv4a))
        pool4 = self.pool4(conv4b)

        conv5a = self.relu(self.conv5a(pool4))
        conv5b = self.relu(self.conv5b(conv5a))
        pool5 = self.pool5(conv5b)
        
        #print(pool5.size())
        flat = torch.flatten(pool5,1)

        fc6 = self.relu(self.fc6(flat))
        fc6 = self.dropout(fc6)

        fc7 = self.relu(self.fc7(fc6))
        fc7 = self.dropout(fc7)

        fc8 = self.fc8(fc7)
        #probs = self.softmax(fc8)

        return fc8

class t2CC3D(nn.Module):
    def __init__(self, num_classes=2):
        super(t2CC3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=(1,1,1)) 
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # 64 * 16 * 16

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 8 * 8

        self.conv3a = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3b = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 4 * 4

        self.conv4a = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv4b = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 2 * 2

        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        conv3a = self.relu(self.conv3a(pool2))
        conv3b = self.relu(self.conv3b(conv3a))
        pool3 = self.pool3(conv3b)

        conv4a = self.relu(self.conv4a(pool3))
        conv4b = self.relu(self.conv4b(conv4a))
        pool4 = self.pool4(conv4b)

        flat = torch.flatten(pool4,1)
        fc5 = self.relu(self.fc5(flat))
        fc5 = self.dropout(fc5)

        fc6 = self.relu(self.fc6(fc5))

        return fc6

class vgg_3D(nn.Module):
    def __init__(self, num_classes=2):
        super(vgg_3D, self).__init__()

        self.conv1_1 = nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=1, padding=(1,1,1)) 
        self.conv1_2 = nn.Conv3d(16, 16, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # 64 * 16 * 16

        self.conv2_1 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 8 * 8

        self.conv3_1 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.conv3_2 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        #self.conv3_3 = nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 4 * 4

        self.conv4_1 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.conv4_2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        #self.conv4_3 = nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) # 128 * 2 * 2

        self.conv5_1 = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.conv5_2 = nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        #self.conv5_3 = nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1,1,1))

        self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.fc6 = nn.Linear(128 * 9 * 4 * 4, 256)
        self.fc7 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        #self.softmax = nn.Softmax()
    
    def forward(self, x):
        #print(x.size())
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        #conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        #conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        #conv5_3 = self.relu(self.conv5_3(conv5_2))
        pool5 = self.pool5(conv5_2)
        
        #print(pool5.size())
        flat = torch.flatten(pool5,1)

        fc6 = self.relu(self.fc6(flat))
        fc6 = self.dropout(fc6)

        fc7 = self.fc7(fc6)
        #fc7 = self.relu(self.fc7(fc6))

        #fc7 = self.fc8(fc6)
        #probs = self.softmax(fc8)

        return fc7