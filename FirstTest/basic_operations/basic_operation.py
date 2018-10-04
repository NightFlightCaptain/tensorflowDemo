# -*- coding:utf-8 -*-
# __author__ = 'wanhaoran'

import tensorflow as tf

# a = tf.placeholder(tf.int32)
# b = tf.placeholder(tf.int32)
#
# add = tf.add(a,b)
# multiply = tf.multiply(a,b)
#
# with tf.Session() as sess:
#     print("add: %i" % sess.run(add, feed_dict={a: 7, b: 9}))
#     print("mul: %i" % sess.run(multiply, feed_dict={a: 7, b: 9}))
#
# matrix1 = tf.constant([[3.,7.]])
# matrix2 = tf.constant([[2.], [9.]])
# product1 = tf.multiply(matrix1, matrix2)
# product2 = tf.matmul(matrix1,matrix2)
#
# print(matrix1,"  ",matrix2)
# with tf.Session() as sess:
#     result1 = sess.run(product1)
#     result2 = sess.run(product2)
#     print(result1,"\n\n",result2,"\n\n",sess.run(tf.multiply(matrix2,matrix1)))

x = [[1, 1, 1], [1, 1, 1]]
y = tf.constant(x)
# tf.reduce_sum(x)
# tf.reduce_sum(x, 0)
# tf.reduce_sum(x, 1)
# tf.reduce_sum(x, 1, keep_dims=True)
# tf.reduce_sum(x, [0, 1])
sess = tf.Session()
print(y)
print(sess.run(tf.reduce_sum(y)))
