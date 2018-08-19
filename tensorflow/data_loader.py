"""
CUDA_VISIBLE_DEVICES=0 python -i mnist_classification.py \
#--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import os
#import cv2 as cv
import numpy as np
# from PIL import Image as im
import imageio as im
from classifier import config
if config.nsml:
	import nsml
	from nsml import DATASET_PATH


# extensions = ('.jpg', '.png')
# /shared/data/mnist_png/train/0/1.png

# fine_list: str, list, bool, bool -> list of str
# Find the file list recursively 
def file_list(path, extensions, sort=True, path_label = False):
	if path_label == True:
		result = [(os.path.join(dp, f) + ' ' + os.path.join(dp, f).split('/')[-2])
		for dp, dn, filenames in os.walk(path) 
		for f in filenames if os.path.splitext(f)[1] in extensions]
	else:
		result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) 
		for f in filenames if os.path.splitext(f)[1] in extensions]
	if sort:
		result.sort()

	return result



def make_list_file(path, save_path, extensions, path_label = False, iter = 1):
	# make the save dir if it is not exists
	#save_path = os.path.join(path, 'meta')
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	print('Finding all input files...')
	file_lst = file_list(path, extensions, True, path_label)
	lenth = len(file_lst)

	print('Writing input file list...')
	for itr in range(iter):
		# save the file inside of the meta/ folder
		f = open(os.path.join(save_path, 'path_label_list{0:03d}.txt'.format(itr)), 'w')
		for line in file_lst[int((itr)*lenth/iter):int((itr+1)*lenth/iter)]:
			f.write(line + '\n')
		f.close()

	print('Listing completed...')



im_size = [28, 28]
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
		img = np.asarray(skio.imread(impath))
		im_batch[i] = img

	if norm == True : 
		im_batch /= 256
	if convert == 'rgb2gray':
		im_batch = np.mean(im_batch, axis=3)

	# Label processing 
	n_classes = len(label_list)
	label_indices = np.array([input_labels]).reshape(-1)
	one_hot_labels = np.eye(n_classes)[label_indices]

	return im_batch, one_hot_labels, gt_labels

