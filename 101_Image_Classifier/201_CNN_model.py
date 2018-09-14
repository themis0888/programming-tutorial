"""
CUDA_VISIBLE_DEVICES=0 python -i 101_Training.py \
--lable_processed True \

"""
import tensorflow as tf
import argparse
import os, random
import data_loader
import numpy as np
import pdb


# -------------------- Model -------------------- #
class CNN_model():
	def __init__(self, sess, config, name):
		self.sess = sess
		self.name = name
		self._build_net()
		self.training = tf.placeholder(tf.bool)

	def _build_net(self):		
		depth = 3
		window = 3
		height = 28
		width = 28
		channels = 3
		filt = [32, 64]
		im_size = [height, width, channels]

		self.X = tf.placeholder(tf.float32, [None, 28, 28, 3])
		self.Y = tf.placeholder(tf.float32, [None, 10])

		self.input_layer = tf.reshape(self.X, [-1, 28, 28, 3])

		# Convolutional Layer #1
		self.conv = self.input_layer
		for i in range(3):
			self.conv = tf.layers.conv2d(inputs=self.conv, filters=filt[0],
				kernel_size=[window, window], padding="same", activation=tf.nn.relu)

		# Pooling Layer #1
		self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=[2, 2], strides=2)

		# Convolutional Layer #2 and Pooling Layer #2
		for i in range(3):
			self.conv = tf.layers.conv2d(inputs=self.pool, filters=filt[1],
				kernel_size=[window, window], padding="same", activation=tf.nn.relu)

		# Pooling Layer #2
		self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=[2, 2], strides=2)

		# Dense Layer
		self.pool2_flat = tf.reshape(self.pool, [-1, 7 * 7 * filt[1]])
		self.dense = tf.layers.dense(inputs=self.pool2_flat, units=1024, activation=tf.nn.relu)
		#dropout = tf.layers.dropout(inputs=dense, rate=0.4)

		# Logits Layer
		self.logits = tf.layers.dense(inputs=self.dense, units=10)


		# -------------------- Objective -------------------- #

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y), name='Loss')
		total_var = tf.global_variables() 
		self.optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.01).minimize(self.cost)
		self.is_correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

		#accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(tf.nn.sigmoid(self.logits)) - tf.round(Y)))
		# accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(self.logits) - tf.round(Y)))
		self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

		self.writer = tf.summary.FileWriter("./board/sample", self.sess.graph)
		self.acc_hist = tf.summary.scalar("Training_accuracy", self.accuracy)
		self.merged = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.saver = tf.train.Saver(total_var)

		tf.train.start_queue_runners(sess=self.sess)

	def train(self, x_data, y_data, training=True):
		return self.sess.run([self.optimizer, self.cost, self.merged, self.accuracy], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

	def get_accuracy(self, x_test, y_test, training=False):
		return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

	def predict(self, x_data, training=False):
		return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_data, self.training: training})

