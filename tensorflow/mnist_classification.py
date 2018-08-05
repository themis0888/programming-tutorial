"""
CUDA_VISIBLE_DEVICES=0 python -i mnist_classification.py \
--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import nsml
from nsml import DATASET_PATH
import os
import data_loader

from tensorflow.examples.tutorials.mnist import input_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/mnist_png/')
parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=100)
config, unparsed = parser.parse_known_args() 

sess = tf.InteractiveSession()

# -------------------- Model -------------------- #

depth = 7
input_shape = 784
X = tf.placeholder(tf.float32, [None, 28, 28, 3])
Y = tf.placeholder(tf.float32, [None, 10])

input_shape = 784

input_layer = tf.reshape(X, [-1, 28*28*3]) 
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

list_file = list_files[0]
batch_size = config.batch_size
with open(list_file) as f:
	for num_file, l in enumerate(f):
		pass

print('Number of input files: \t{}'.format(num_file))
total_batch = int(num_file / batch_size)
saver = tf.train.Saver()

tf.train.start_queue_runners(sess=sess)

for epoch in range(15):
	total_cost = 0
	Xbatch, Ybatch = data_loader.queue_data(
			list_file, config.n_classes, config.batch_size, 'train', multi_label=False)

	for i in range(total_batch):
		# Xbatch and Ybatch are tensor, which is not feedable to the placeholder.
		print('Data loading')
		#xbatch, ybatch = sess.run([Xbatch, Ybatch])
		print('Hello :D I am going through the {}th iteration'.format(i))
		_, cost_val = sess.run([optimizer, cost], feed_dict={X: Xbatch, Y: Ybatch})
		total_cost += cost_val

	print('Epoch:', '%04d' % (epoch + 1),
		'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

	if epoch % 5 == 0:
		if not os.path.exists('{0:03d}_epoch_model'.format(epoch)):
			os.mkdir('{0:03d}_epoch_model'.format(epoch))
		saver.save(sess, '{0:03d}_epoch_model'.format(epoch))


# -------------------- Testing -------------------- #

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
X, Y = data_loader.queue_data(
			list_file, config.n_classes, config.batch_size, 'val', multi_label=False)

accuracy = sess.run(accuracy, feed_dict = {X: mnist.test.images,
								Y: mnist.test.labels})
print('Accuracy:', accuracy)

