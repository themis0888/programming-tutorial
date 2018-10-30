"""
CUDA_VISIBLE_DEVICES=3 CUDA_CACHE_PATH="/gpu_cache/" \
python feature_extractor.py \
--data_path=/shared/data/sample/ \
--list_path=/shared/data/sample/meta/ \
--model_name=vgg_19


CUDA_VISIBLE_DEVICES=1 python feature_extractor.py \
--data_path=/shared/data/danbooru2017/256px/ \
--list_path=/shared/data/danbooru2017/256px/meta/ \
--model_name=vgg_19
"""

import pdb
import random, time, os, sys
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import scipy.io as sio
import skimage.io as skio
import skimage 
import skimage.transform as sktr

import argparse

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

def lost_file_remover(path, exists_list):
	filenames = file_list(path, ['.jpg'])
	for filename in filenames:
		if not os.path.basename(filename) in exists_list:
			os.remove(filename)

# '/shared/data/f2c_4dcyc/trainA/np_008083.npy'
# '/shared/data/face2cartoon/trainA/008083.jpg'
def convert_npy_f2d(path):
	filenames = file_list(path, ['.npy'])
	counter = 0
	num_tot = len(filenames)
	for filename in filenames:
		counter += 1
		npy_file = np.load(filename)
		[h, w, c] = npy_file.shape
		for y in range(h):
			for x in range(w):
				if npy_file[y, x, -1] != 0:
					npy_file[y, x, -1] = 127
		npy_file[:,:,-1] += 127
		npy_file = np.uint8(npy_file)
		np.save(filename, npy_file)
		if np.mod(counter, 10000):
			print('{0:.3}% done'.format(100*counter/num_tot))


def convert_npy_RGB_BGR(path):
	filenames = file_list(path, ['.npy'])
	for filename in filenames:
		npy_file = np.load(filename)
		[h, w, c] = npy_file.shape
		temp = np.zeros([h, w, c])
		temp[:,:,0] = npy_file[:,:,2]
		temp[:,:,1] = npy_file[:,:,1]
		temp[:,:,2] = npy_file[:,:,0]
		temp[:,:,3] = npy_file[:,:,3]					
	
		np.save(filename, temp)


def mat_resize(npy_file, fine_size):
    img = npy_file
    img_layer = []
    for i in range(img.shape[-1]):
        img_layer.append(np.zeros([fine_size, fine_size]))
        img_layer[i] = sktr.resize(img[:,:,i], [fine_size, fine_size])
        img_layer[i] = np.expand_dims(img_layer[i], axis = -1)
        if i == 0:
            img_concat = img_layer[i]
        else:
            img_concat = np.concatenate((img_concat, img_layer[i]), axis = -1)
    return img_concat


def image_resize(data_path, save_path, input_size, output_size):
	img = skio.imread(data_path)
	img = mat_resize(img, output_size)
	img_name = os.path.basename(data_path)
	skio.imsave(os.path.join(save_path, img_name), img)
	