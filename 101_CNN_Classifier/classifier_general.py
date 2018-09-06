"""
CUDA_VISIBLE_DEVICES=5 python -i classifier_vgg.py \
--data_path=/shared/data/celeb_cartoon/anime/ \
--n_classes=100 --lable_processed True \
--list_path=. --load_checkpoint True
"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/mnist_png/')
parser.add_argument('--list_path', type=str, dest='list_path', default='/shared/data/mnist_png/meta/')
parser.add_argument('--model_name', type=str, dest='model_name', default='vgg_19')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--epoch', type=int, dest='epoch', default=1000)

parser.add_argument('--n_classes', type=int, dest='n_classes', default=100)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=20)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.96)
parser.add_argument('--lable_processed', type=bool, dest='lable_processed', default=True)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=1000)
parser.add_argument('--print_freq', type=int, dest='print_freq', default=50)

parser.add_argument('--mode', type=str, dest='mode', default='pretrained')
parser.add_argument('--load_checkpoint', type=bool, dest='load_checkpoint', default=False)
parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
parser.add_argument('--nsml', type=bool, dest='nsml', default=False)
config, unparsed = parser.parse_known_args() 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import os, random
import data_loader
import numpy as np
import tensorflow.contrib.slim.nets as nets
import pdb


# -------------------- Model -------------------- #

slim = tf.contrib.slim
vgg = nets.vgg

height = 224
width = 224
channels = 3
im_size = [height, width, channels]

X = tf.placeholder(tf.float32, shape=[None] + im_size)
Y = tf.placeholder(tf.float32, [None, config.n_classes])

if config.model_name == 'vgg_19':

	with slim.arg_scope(vgg.vgg_arg_scope()):
		logits, endpoints = vgg.vgg_19(x, num_classes=config.n_classes, is_training=False)
		feat_layer = endpoints['vgg_19/fc7']
	all_vars = tf.all_variables()
	var_to_restore = [v for v in all_vars if not v.name.startswith('vgg_19/fc8')]


elif config.model_name == 'resnet_v1_50':
	res = nets.resnet_v1
	with slim.arg_scope(res.resnet_arg_scope()):
		logits, endpoints = res.resnet_v1_50(x, num_classes=config.n_classes, is_training=False)
		feat_layer = endpoints['resnet_v1_50/block4/unit_3/bottleneck_v1']
	all_vars = tf.all_variables()
	var_to_restore = [v for v in all_vars] # if not v.name.startswith('predictions')]


elif config.model_name == 'inception_v2':
	with slim.arg_scope(inc.inception_v2_arg_scope()):
		logits, endpoints = inc.inception_v2(x, num_classes=config.n_classes, is_training=False)
		feat_layer = endpoints['PreLogits']
	all_vars = tf.all_variables()
	var_to_restore = [v for v in all_vars] 

feat_layer = tf.reshape(feat_layer, [-1, 4096])
# Output logits Layer
logits = tf.layers.dense(inputs=feat_layer, units=config.n_classes, activation=None, name='Logit')


# -------------------- Objective -------------------- #

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y), name='Loss')
total_var = tf.global_variables() 
optimizer_1 = tf.train.AdamOptimizer(0.001, epsilon=0.01).minimize(cost, var_list= [v for v in total_var if not v in var_to_restore]) 
optimizer_2 = tf.train.AdamOptimizer(0.001, epsilon=0.01).minimize(cost)
#is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(tf.nn.sigmoid(logits)) - tf.round(Y)))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

writer = tf.summary.FileWriter("./board/sample", sess.graph)
acc_hist = tf.summary.scalar("Training accuracy", accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess.run(init)

tf.train.start_queue_runners(sess=sess)

print('Mode: {}'.format(config.mode))
if config.load_checkpoint:
	saver = tf.train.import_meta_graph(os.path.join(
		config.checkpoint_path, 'fc_network_000.meta'))
	saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_path))

	# saver = tf.train.Saver(total_var)
	# saver.restore(sess, os.path.join(config.checkpoint_path))
	print('Trained model Load Success')
else:
	# saver = tf.train.Saver(var_to_restore)
	# saver.restore(sess, os.path.join(config.model_path, 'vgg_19.ckpt'))
	print('VGG_19 pretrained model Loaded')


# -------------------- Data maniging -------------------- #

label_file = np.load(os.path.join('/shared/data/danbooru2017/256px/','attributes.npz'))

data_loader.make_dict_file(config.data_path, config.list_path, 
	label_file['attributes'], ('.png', '.jpg'), False, 1)
list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.list_path) 
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
					path_label_dict[train_data[i*batch_size + j+1]], 
					axis = -1)), axis = -1)
			Ybatch = np.reshape(label_list, [batch_size, config.n_classes])

			Xbatch = data_loader.queue_data_dict(
				train_data[i*batch_size:(i+1)*batch_size], im_size, config.lable_processed)

			# pdb.set_trace()
			if counter < 200:
				_, cost_val, acc, acc_ = sess.run([optimizer_1, cost, merged, accuracy], 
					feed_dict={X: Xbatch, Y: Ybatch})
			else:
				_, cost_val, acc, acc_ = sess.run([optimizer_2, cost, merged, accuracy], 
					feed_dict={X: Xbatch, Y: Ybatch})
			total_cost += cost_val

			counter += 1

			if np.mod(counter, config.print_freq) == 0:
				print('Step:', '%05dk' % (counter),
					'\tAvg. cost =', '{:.5f}'.format(cost_val),
					'\tAcc: {:.5f}'.format(acc_))
				writer.add_summary(acc, counter)

			# Save the model
			if np.mod(counter, config.save_freq) == 0:
				if config.nsml:
					nsml.save(counter)
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

