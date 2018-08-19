"""
CUDA_VISIBLE_DEVICES=0 python -i classifier.py \
--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/mnist_png/')
parser.add_argument('--list_path', type=str, dest='list_path', default='/shared/data/mnist_png/meta/')
parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=100)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.96)

parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
parser.add_argument('--nsml', type=bool, dest='nsml', default=False)
config, unparsed = parser.parse_known_args() 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import os, random
import data_loader
import numpy as np
import tensorflow.contrib.slim.nets as nets


# -------------------- Model -------------------- #

slim = tf.contrib.slim
vgg = nets.vgg

height = 224
width = 224
channels = 3
im_size = [height, width, channels]

X = tf.placeholder(tf.float32, shape=[None] + im_size)
Y = tf.placeholder(tf.float32, [None, 10])
with slim.arg_scope(vgg.vgg_arg_scope()):
	logits, end_points = vgg.vgg_19(X, num_classes=1000, is_training=False)
	feat_layer = end_points['vgg_19/fc7']
	all_vars = tf.all_variables()
	var_to_restore = [v for v in all_vars]

feat_layer = tf.reshape(feat_layer, [-1, 4096])
# Output logits Layer
logits = tf.layers.dense(inputs= feat_layer, units=10)


# -------------------- Objective -------------------- #

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

writer = tf.summary.FileWriter("./board/sample", sess.graph)
acc_hist = tf.summary.scalar("Training accuracy", accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver(var_to_restore)
saver.restore(sess, "/shared/data/models/vgg_19.ckpt")


# -------------------- Data maniging -------------------- #

data_loader.make_list_file(config.data_path, config.list_path, ('.png', '.jpg'), True, 1)
list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.list_path) 
		for f in filenames if 'path_label_list' in f]
list_files.sort()

batch_size = config.batch_size

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
				train_data[i*batch_size:(i+1)*batch_size], label_list)
	
			_, cost_val, acc = sess.run([optimizer, cost, merged], feed_dict={X: Xbatch, Y: Ybatch})
			total_cost += cost_val

			if np.mod(i, 10) == 0:
				print('Epoch:', '%04d' % (epoch + 1),
					'\tAvg. cost =', '{:.3f}'.format(total_cost / total_batch))

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

