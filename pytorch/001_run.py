"""
CUDA_VISIBLE_DEVICES=0 python -i 001_run.py \
--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import os
import numpy as np
import skimage.io as skio
import scipy.misc
import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
	
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_feature(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		middle = x
		x = self.fc3(x)
		return x, middle
	
	def num_flat_feature(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
	
	def calculate_loss(self, output, target):
		
		self.optimizer = optim.SGD(self.parameters(), lr = 0.01)
		self.optimizer.zero_grad()
		criterion = nn.MSELoss()
		self.loss = criterion(output, target)
		self.optimizer.step()


class Classification:
	
	def __init__(self, args = None):
		self.net = Net()

	def training(self, input):
		self.net.zero_grad()
		self.output = self.net(input)
		self.target = torch.randn(10)
		self.target = self.target.view(1, -1)
		
		self.net.calculate_loss(self.output, self.target)
		"""
		criterion = nn.MSELoss()	#torch.abs(torch.mean(self.target - self.output))
		
		self.optimizer = optim.SGD(self.net.parameters(), lr = 0.01)
		self.optimizer.zero_grad()

		self.loss = criterion(self.output, self.target)
		self.loss.backward()
		self.optimizer.step()
		"""
		

model = Classification()
input = torch.randn([1,1,32,32])
step = 10000
period = 100

for i in range(step):

	model.training(input)
	if i % period == 0:
		print(model.net.loss)


	


		