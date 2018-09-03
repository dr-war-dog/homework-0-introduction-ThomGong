#encoding = UTF-8
### Author: Tong Gong
### Course: DataViz (IS590)
### Date: 09/02/2018

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# create tensorflow structure-------------------------------------------------------------------------
Weights = tf.Variable(tf.random_uniform([1], -1.0 , 1.0))
Biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + Biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initializing
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Biases))

sess.close()



