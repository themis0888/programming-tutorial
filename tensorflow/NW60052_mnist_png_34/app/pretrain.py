from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

height = 299
width = 299
channels = 3

X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(inception_resnet_v2_arg_scope()):
     logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "/home/pramod/Downloads/inception_resnet_v2_2016_08_30.ckpt")
