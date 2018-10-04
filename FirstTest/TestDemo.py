# -*- coding:utf-8 -*-
# __author__ = 'wanhaoran'

import tensorflow as tf
import numpy as np


def test_first():
    '''
    手动循环来进行梯度训练
    :return:
    '''
    W = tf.Variable([.1], dtype=tf.float32, name='W')
    b = tf.Variable([-.1], dtype=tf.float32, name='b')

    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

    linear_model = W * x + b

    with tf.name_scope("loss-model"):
        loss = tf.reduce_sum(tf.square(linear_model - y))
        tf.summary.scalar("loss", loss)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    # 训练数据集
    x_train = [1, 2, 3, 6, 8]
    y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/tensorflow', sess.graph)

    for i in range(10000):
        summary, _ = sess.run([merged, train], {x: x_train, y: y_train})
        writer.add_summary(summary, i)

    curr_W, curr_b, curr_loss = sess.run(
        [W, b, loss], {x: x_train, y: y_train}
    )

    print('W: %s b: %s loss: %s' % (curr_W, curr_b, curr_loss))


def test_second():
    '''
    这个函数是使用tensorflow中自带的LinearRegressor线性回归

    :return: null
    '''
    # 保存训练用的数据集
    x_train = np.array([1., 2., 3., 6., 8.])
    y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

    x_eavl = np.array([2., 5., 7., 9.])
    y_eavl = np.array([7.6, 17.2, 23.6, 28.8])

    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
    #   LinearRegressor训练器
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True
    )
    # 再用训练数据创建一个输入模型，用来进行后面的模型评估
    train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False
    )

    # 用评估数据创建一个输入模型，用来进行后面的模型评估
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_eavl}, y_eavl, batch_size=2, num_epochs=1000, shuffle=False
    )

    # 使用训练数据训练1000次
    estimator.train(input_fn=train_input_fn, steps=1000)

    train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
    print("train metrics: %r" % train_metrics)

    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("eval metrics: %r" % eval_metrics)


def model_fn(features, labels, mode):
    # 构建线性模型
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # 构建损失模型
    loss = tf.reduce_sum(tf.square(y - labels))
    # 训练模型子图
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    # 通过EstimatorSpec指定我们的训练子图积极损失模型
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train
    )


def test_third():
    '''

    :return:null
    '''
    estimator = tf.estimator.Estimator(model_fn=model_fn)

    # 后面的训练逻辑与使用LinearRegressor一样
    x_train = np.array([1., 2., 3., 6., 8.])
    y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

    x_eavl = np.array([2., 5., 7., 9.])
    y_eavl = np.array([7.6, 17.2, 23.6, 28.8])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True)

    train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_eavl}, y_eavl, batch_size=2, num_epochs=1000, shuffle=False)

    estimator.train(input_fn=train_input_fn, steps=1000)

    train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
    print("train metrics: %r" % train_metrics)

    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("eval metrics: %s" % eval_metrics)


test_first()
