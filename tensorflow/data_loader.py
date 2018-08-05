"""
CUDA_VISIBLE_DEVICES=0 python -i mnist_classification.py \
--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import nsml
from nsml import DATASET_PATH
import os
import cv2 as cv
import numpy as np

im_size = [28, 28, 3]
# queue_data(lst, ['0', '1', '2'], norm=True, convert = 'rgb2gray')
# queue_data does not consider the batch size but return the all data on the list.
def queue_data(file_list, label_list, norm=True, convert = None):
	# Batch frame fit into the image size 
	batch_size = len(file_list)
	im_batch = np.zeros([batch_size] + im_size)
	input_labels = []
	gt_labels = []

	# Reading from the list
	for i in range(batch_size):
		impath, input_label = file_list[i].split(' ')
		input_label.replace('\n', '')
		# return the index of the label 
		gt_labels.append(input_label)
		input_labels.append(label_list.index(input_label))
		im = cv.imread(impath)
		im_batch[i] = im

	if norm == True : 
		im_batch /= 256
	if convert == 'rgb2gray':
		im_batch = np.mean(im_batch, axis=3)

	# Label processing 
	n_classes = len(label_list)
	label_indices = np.array([input_labels]).reshape(-1)
	one_hot_labels = np.eye(n_classes)[label_indices]

	return im_batch, one_hot_labels, gt_labels

