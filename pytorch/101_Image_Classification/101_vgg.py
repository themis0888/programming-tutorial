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
import vgg



class Classification:
	
	def __init__(self, args = None):
		self.net = Net()

	def training(self, input):
		self.net.zero_grad()
		self.output = self.net(input)
		self.target = torch.randn(10)
		self.target = self.target.view(1, -1)
		
		self.net.calculate_loss(self.output, self.target)
		criterion = nn.MSELoss()	#torch.abs(torch.mean(self.target - self.output))
		
		self.optimizer = optim.SGD(self.net.parameters(), lr = 0.01)
		self.optimizer.zero_grad()

		self.loss = criterion(self.output, self.target)
		self.loss.backward()
		self.optimizer.step()
		

model = Classification()
input = torch.randn([1,1,32,32])
step = 10000
period = 100

for i in range(step):

	model.training(input)
	if i % period == 0:
		print(model.net.loss)


	


		