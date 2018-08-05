"""
CUDA_VISIBLE_DEVICES=0 python -i mnist_classification.py \
--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import nsml
from nsml import DATASET_PATH
import os, random
import data_loader
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/mnist_png/')
parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=100)
config, unparsed = parser.parse_known_args() 

sess = tf.InteractiveSession()

# -------------------- Model -------------------- #

X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])

input_layer = tf.reshape(X, [-1, 28*28])
input_layer = tf.contrib.layers.flatten(input_layer)

fc = tf.layers.dense(inputs = input_layer, units = 256, activation = tf.nn.relu)
fc = tf.layers.dense(inputs = fc, units = 256, activation = tf.nn.relu)
fc = tf.layers.dense(inputs = fc, units = 128, activation = tf.nn.relu)

# Output logits Layer
logits = tf.layers.dense(inputs= fc, units=10)

# -------------------- Objective -------------------- #

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess.run(init)


# -------------------- Learning -------------------- #

list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(
			os.path.join(config.data_path, 'meta')) 
		for f in filenames if 'path_label_list' in f]
list_files.sort()

batch_size = config.batch_size
saver = tf.train.Saver()

label_list = [str(i) for i in range(config.n_classes)]

for epoch in range(5):
	for list_file in list_files:

		with open(list_file) as f:
			path_label_list = f.read().split('\n')
			# You should shuffle the list. 
			# The network will be stupid if you don't  
			random.shuffle(path_label_list)
			train_data = [line for line in path_label_list
			if 'train' in line]
			test_data = [line for line in path_label_list
			if 'test' in line]

			num_file = len(train_data)

		# print('Number of input files: \t{}'.format(num_file))
		total_batch = int(num_file / batch_size)
		total_cost = 0

		for i in range(total_batch):
			# Get the batch as [batch_size, 28,28] and [batch_size, n_classes] ndarray
			Xbatch, Ybatch, _ = data_loader.queue_data(
				train_data[i*batch_size:(i+1)*batch_size], label_list, convert = 'rgb2gray')
	
			_, cost_val = sess.run([optimizer, cost], feed_dict={X: Xbatch, Y: Ybatch})
			total_cost += cost_val

	print('Epoch:', '%04d' % (epoch + 1),
		'\tAvg. cost =', '{:.3f}'.format(total_cost / total_batch))

	"""
	if epoch % 5 == 0:
		if not os.path.exists('{0:03d}_epoch_model'.format(epoch)):
			os.mkdir('{0:03d}_epoch_model'.format(epoch))
		saver.save(sess, '{0:03d}_epoch_model'.format(epoch))
	"""

# -------------------- Testing -------------------- #


is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
Xbatch, Ybatch, _ = data_loader.queue_data(
	test_data, label_list, convert = 'rgb2gray')

accuracy_ = sess.run(accuracy, feed_dict = {X: Xbatch, Y: Ybatch})
print('Accuracy:', accuracy_)

