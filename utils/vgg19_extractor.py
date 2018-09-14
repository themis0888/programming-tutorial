import pdb
import time
import random
import numpy as np
import scipy
import tensorflow as tf
import os
import sys
#from tensorflow.examples.tutorials.mnist import input_data
#import alexnet_model
import tensorflow.contrib.slim.nets as nets
#import mnist_model_sonic as mnist_model
#from scipy.io import loadmat
import data_loader
import scipy.io as sio

flags = tf.app.flags
#################################################
########### model configuration #################
#################################################
flags.DEFINE_float('memory_usage', 0.96, 'GRU memory to use')
#################################################
########## network configuration ################
#################################################
flags.DEFINE_integer('n_classes', 50, 'MNIST dataset')

flags.DEFINE_integer('max_iter', 300000, '')
flags.DEFINE_integer('batch_size', 1, '')

flags.DEFINE_integer('train_display', 200, '')
flags.DEFINE_integer('val_display', 1000, '')
flags.DEFINE_integer('val_iter', 100, '')

#################################################
########## checkpoint configuration #############
#################################################
flags.DEFINE_string('checkpoint_name', '/st1/dhna/sampling/result/checkpoints/imagenet_alexnet_deepset', '')
flags.DEFINE_integer('save_iter', 50000, '')
flags.DEFINE_string('ds_save_path', '/st1/dhna/sampling/result/', '')
flags.DEFINE_string('ds_save_etc', '_d_lr_04_set4_', '')
FLAGS = flags.FLAGS

slim = tf.contrib.slim
vgg = nets.vgg

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# TODO Read imagenet dataset
awa_train_path = '/st1/dhna/sampling/code/sample_net/awa_whole_list.txt'# NOTE needs to be changed

trainX, trainY = data_loader.queue_data(
		awa_train_path, FLAGS.n_classes, FLAGS.batch_size, 'val', multi_label=False)

""" TRAINING """
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes])
keep_prob = tf.placeholder(tf.float32)


# NOTE Main Network
with slim.arg_scope(vgg.vgg_arg_scope()):
	logits, endpoints = vgg.vgg_19(x, num_classes=FLAGS.n_classes, is_training=False)
	feat_fc1 = endpoints['vgg_19/fc7']

all_vars = tf.all_variables()
var_to_restore = [v for v in all_vars if not v.name.startswith('vgg_19/fc8')]

tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver(var_to_restore)
saver.restore(sess, "/st1/mkcho/vgg_19.ckpt")

feat = []
lab = []
for i in range(30475):
	batch_x, batch_y = sess.run([trainX, trainY])
	_, idx = np.nonzero(batch_y)

	feature = sess.run(feat_fc1, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})
	feat.append(feature[0][0][0])
	lab.append(idx[0])
	if i%100 == 0:
		print (i, " th iteration")

sio.savemat('/st1/mkcho/vgg_19_awa.mat', {'feature': feat, 'label': lab})

print('end')
