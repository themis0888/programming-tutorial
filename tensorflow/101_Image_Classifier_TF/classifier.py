"""
CUDA_VISIBLE_DEVICES=0 python -i classifier.py 
"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/mnist_png/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/mnist_png/')
parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=100)

parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
parser.add_argument('--nsml', type=bool, dest='nsml', default=False)
config, unparsed = parser.parse_known_args() 

import os, random
import data_loader
import numpy as np

sess = tf.InteractiveSession()

# -------------------- Model -------------------- #

im_size = [28, 28]

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

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)
# saver.restore(sess, os.path.join(config.checkpoint_path, 'fc_network_{}'.format(10)))
writer = tf.summary.FileWriter("./board/sample", sess.graph)
acc_hist = tf.summary.scalar("Training accuracy", accuracy)
merged = tf.summary.merge_all()

# -------------------- Data maniging -------------------- #


data_loader.make_list_file(config.data_path, config.meta_path, ('.png', '.jpg'), True, 1)
list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.meta_path) 
		for f in filenames if 'path_label_list' in f]
list_files.sort()

batch_size = config.batch_size
saver = tf.train.Saver()

label_list = [str(i) for i in range(config.n_classes)]


# -------------------- Learning -------------------- #

for epoch in range(15):
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
		final_acc = 0

		for i in range(total_batch):
			# Get the batch as [batch_size, 28,28] and [batch_size, n_classes] ndarray
			Xbatch, Ybatch, _ = data_loader.queue_data(
				train_data[i*batch_size:(i+1)*batch_size], label_list,
				im_size)
	
			_, cost_val, acc = sess.run([optimizer, cost, merged], feed_dict={X: Xbatch, Y: Ybatch})
			total_cost += cost_val


	print('Epoch:', '%04d' % (epoch + 1),
		'\tAvg. cost =', '{:.3f}'.format(total_cost / total_batch))

	writer.add_summary(acc, epoch)
	# Save the model
	if epoch % 5 == 0:
		if not os.path.exists(config.checkpoint_path):
			os.mkdir(config.checkpoint_path)
		saver.save(sess, os.path.join(config.checkpoint_path, 
			'fc_network_{0:03d}'.format(epoch)))
	

# -------------------- Testing -------------------- #


Xbatch, Ybatch, _ = data_loader.queue_data(
	test_data, label_list)

accuracy_ = sess.run(accuracy, feed_dict = {X: Xbatch, Y: Ybatch})
print('Accuracy:', accuracy_)

