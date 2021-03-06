"""
CUDA_VISIBLE_DEVICES=0 python -i 001_MNIST_FC_Classifier.py \
--lable_processed True \

"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/mnist_png/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/mnist_png/')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--epoch', type=int, dest='epoch', default=1000)

parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=100)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.96)
parser.add_argument('--lable_processed', type=bool, dest='lable_processed', default=True)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=1000)
parser.add_argument('--print_freq', type=int, dest='print_freq', default=50)

parser.add_argument('--mode', type=str, dest='mode', default='pretrained')
parser.add_argument('--load_checkpoint', type=bool, dest='load_checkpoint', default=False)
parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
config, unparsed = parser.parse_known_args() 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import os, random
import data_loader
import numpy as np
import pdb


# -------------------- Model -------------------- #

depth = 3
window_1 = 3
window_2 = 3
height = 224
width = 224
channels = 3
im_size = [height, width, channels]

X = tf.placeholder(tf.float32, [None, 28, 28, 3])
Y = tf.placeholder(tf.float32, [None, 10])

input_layer = tf.reshape(X, [-1, 28, 28, 3])

# Convolutional Layer #1
conv1_1 = conv_layer(input_layer, 3, 64, "conv1_1")
conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2")
pool1 = max_pool(conv1_2, 'pool1')

conv2_1 = conv_layer(pool1, 64, 128, "conv2_1")
conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2")
pool2 = max_pool(conv2_2, 'pool2')

conv3_1 = conv_layer(pool2, 128, 256, "conv3_1")
conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2")
conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3")
conv3_4 = conv_layer(conv3_3, 256, 256, "conv3_4")
pool3 = max_pool(conv3_4, 'pool3')

conv4_1 = conv_layer(pool3, 256, 512, "conv4_1")
conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2")
conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3")
conv4_4 = conv_layer(conv4_3, 512, 512, "conv4_4")
pool4 = max_pool(conv4_4, 'pool4')

conv5_1 = conv_layer(pool4, 512, 512, "conv5_1")
conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2")
conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3")
conv5_4 = conv_layer(conv5_3, 512, 512, "conv5_4")
pool5 = max_pool(conv5_4, 'pool5')

fc6 = fc_layer(pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
relu6 = tf.nn.relu(fc6)

if train_mode is not None:
	relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(relu6, dropout), lambda: relu6)
elif trainable:
	relu6 = tf.nn.dropout(relu6, dropout)

fc7 = fc_layer(relu6, 4096, 4096, "fc7")
relu7 = tf.nn.relu(fc7)
if train_mode is not None:
	relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(relu7, dropout), lambda: relu7)
elif trainable:
	relu7 = tf.nn.dropout(relu7, dropout)

fc8 = fc_layer(relu7, 4096, 1000, "fc8")

# Logits Layer
logits = tf.layers.dense(inputs=dense, units=10)


# -------------------- Objective -------------------- #

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='Loss')
total_var = tf.global_variables() 
optimizer_1 = tf.train.AdamOptimizer(0.001, epsilon=0.01).minimize(cost)
#is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

#accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(tf.nn.sigmoid(logits)) - tf.round(Y)))
accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(logits) - tf.round(Y)))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

writer = tf.summary.FileWriter("./board/sample", sess.graph)
acc_hist = tf.summary.scalar("Training accuracy", accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(total_var)

tf.train.start_queue_runners(sess=sess)


# -------------------- Data maniging -------------------- #

#label_file = np.load(os.path.join('/shared/data/celeb_cartoon/','attributes.npz'))
label_file = np.load(os.path.join(config.meta_path, 'path_label_dict.npy'))

list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.meta_path) 
		for f in filenames if 'dict.npy' in f]
list_files.sort()

batch_size = config.batch_size


# -------------------- Training -------------------- #

# feat, x, y = sess.run([feat_layer,logits, Y], feed_dict = {X: Xbatch, Y: Ybatch})
# _ = sess.run(optimizer, feed_dict = {X: Xbatch, Y: Ybatch})

counter = 0
for epoch in range(config.epoch):
	for list_file in list_files:

		f = np.load(list_file)
		
		path_label_dict = f.item()
		input_file_list = list(path_label_dict.keys())
		# You should shuffle the list. 
		# The network will be stupid if you don't
		random.shuffle(input_file_list)
		all_data = [line for line in input_file_list]
		train_data = all_data[1000:]
		test_data = all_data[:1000]

		num_file = len(train_data)

		if num_file ==0:
			break

		# print('Number of input files: \t{}'.format(num_file))
		total_batch = int(num_file / batch_size)
		total_cost = 0
		final_acc = 0

		for i in range(total_batch):
			# Get the batch as [batch_size, 28,28] and [batch_size, n_classes] ndarray
			label_list = np.expand_dims(path_label_dict[train_data[i*batch_size]], axis = -1)
			for j in range(1, batch_size):
				label_list = np.concatenate((label_list, np.expand_dims(
					path_label_dict[train_data[i*batch_size + j]], 
					axis = -1)), axis = -1)
			Ybatch = np.reshape(label_list, [batch_size, config.n_classes])

			Xbatch = data_loader.queue_data_dict(
				train_data[i*batch_size:(i+1)*batch_size], im_size, config.lable_processed)

			_, cost_val, acc, acc_ = sess.run([optimizer_1, cost, merged, accuracy], feed_dict={X: Xbatch, Y: Ybatch})
			

			total_cost += cost_val

			counter += 1

			if np.mod(counter, config.print_freq) == 0:
				print('Step:', '%05dk' % (counter),
					'\tAvg. cost =', '{:.5f}'.format(cost_val),
					'\tAcc: {:.5f}'.format(acc_))
				writer.add_summary(acc, counter)

			# Save the model
			if np.mod(counter, config.save_freq) == 0:
				if not os.path.exists(config.checkpoint_path):
					os.mkdir(config.checkpoint_path)
				saver.save(sess, os.path.join(config.checkpoint_path, 
					'vgg19_{0:03d}k'.format(int(counter/1000))))
				print('Model ')
	

# -------------------- Testing -------------------- #


Xbatch, Ybatch, _ = data_loader.queue_data(
	test_data, label_list, im_size)

accuracy_ = sess.run(accuracy, feed_dict = {X: Xbatch, Y: Ybatch})
print('Accuracy:', accuracy_)
