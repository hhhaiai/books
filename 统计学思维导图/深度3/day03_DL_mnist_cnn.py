import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
# 利用数据，在训练的时候实时提供数据
# mnist手写数字数据在运行时候实时提供给给占位符
tf.app.flags.DEFINE_integer("is_train", 1, "指定是否是训练模型，还是拿数据去预测")
FLAGS = tf.app.flags.FLAGS


def create_variable(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.01))

def create_cnn_model(x):

    with tf.variable_scope("conv1"):
        # 卷积层
        # 调整x的形状
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # conv1_weights = tf.Variable(tf.random_normal(shape=[5, 5, 1, 32]))
        conv1_weights = create_variable([5, 5, 1, 32])
        # conv1_bias = tf.Variable(tf.random_normal(shape=[32]))
        conv1_bias = create_variable([32])
        conv1_tensor = tf.nn.conv2d(x_reshape, conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_tensor = tf.nn.relu(conv1_tensor)

        # 池化层
        pool1_tensor = tf.nn.max_pool(relu1_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("conv2"):
        # 卷积层
        # conv1_weights = tf.Variable(tf.random_normal(shape=[5, 5, 1, 32]))
        conv2_weights = create_variable([5, 5, 32, 64])
        # conv1_bias = tf.Variable(tf.random_normal(shape=[32]))
        conv2_bias = create_variable([64])
        conv2_tensor = tf.nn.conv2d(pool1_tensor, conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_tensor = tf.nn.relu(conv2_tensor)

        # 池化层
        pool2_tensor = tf.nn.max_pool(relu2_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("fc"):
        # [-1, 7 * 7 * 64] * [7 * 7 * 64, 10] = [None, 10]
        fc_x = tf.reshape(pool2_tensor, [-1, 7 * 7 * 64])

        fc_weights = create_variable([7 * 7 * 64, 10])
        fc_bias = create_variable([10])
        #
        # # y_predict (None, 10)
        y_predict = tf.matmul(fc_x, fc_weights) + fc_bias

    return y_predict

    # 第一个卷积大层
    # 卷积：
    # 32
    # 个filter
    # 5 * 5
    # 步长1 “SAME”
    # filter：指定过滤器的权重数量，[filter_height, filter_width, in_channels, out_channels]
    # strides：strides = [1, stride, stride, 1], 步长

    # [-1, 28, 28, 1] --> [-1, 28, 28, 32]
    # 激活：
    # Relu
    # relu1_tensor = tf.nn.relu(conv1_tensor)
    # 池化：
    # filter
    # 2 * 2
    # 步长2
    # pool1_tensor = tf.nn.max_pool(relu1_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # [-1, 28, 28, 32] --> [-1, 14, 14, 32]
    # 第二个卷积大层：
    # 卷积：
    # 64
    # 个filter
    # 5 * 5
    # 步长1 “SAME”
    # filter：指定过滤器的权重数量，[filter_height, filter_width, in_channels, out_channels]
    # conv2_weights = tf.Variable(tf.random_normal(shape=[5, 5, 32, 64]))
    # conv2_bias = tf.Variable(tf.random_normal(shape=[64]))
    # conv2_tensor = tf.nn.conv2d(pool1_tensor, conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias
    # [-1, 14, 14, 32] --> [-1, 14, 14, 64]
    # 激活：
    # Relu
    # relu2_tensor = tf.nn.relu(conv2_tensor)
    # 池化：
    # filter
    # 2 * 2
    # 步长2
    # pool2_tensor = tf.nn.max_pool(relu2_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # [-1, 14, 14, 64] --> [-1, 7, 7, 64]
    # 第三层：全连接层：
    # [-1, 7 * 7 * 64] * [7 * 7 * 64, 10] = [None, 10]
    # fc_x = tr.reshape(pool2_tensor, [-1, 7 * 7(64)])
    # fc_weights = tf.Variable(tf.random_normal(shape=[7 * 7 * 64, 10]))
    # fc_bias = tf.Variable(tf.random_normal(shape=[710]))
    #
    # # y_predict (None, 10)
    # y_predict = tf.matmul(fc_x, fc_weights) + fc_bias


def full_connection():
    """
    单层全连接神经网络识别手写数字图片
    特征值：[None, 784]
    目标值：one_hot编码 [None, 10]
    :return:
    """
    # 1、准备数据
    # x [None, 784] y_true [None. 10]
    with tf.variable_scope("prepare_data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])
    # 2、全连接层神经网络计算
    # 类别：10个类别  全连接层：10个神经元
    # 参数w: [784, 10]   b:[10]
    # 全连接层神经网络的计算公式：[None, 784] * [784, 10] + [10] = [None, 10]
    # 随机初始化权重偏置参数，这些是优化的参数，必须使用变量op去定义
    # with tf.variable_scope("fc_model"):
    #     weight = tf.Variable(tf.random_normal([784, 10]), name="w")
    #     bias = tf.Variable(tf.random_normal([10]), name="b")
    #     # fc层的计算
    #     # y_predict [None, 10]输出结果，提供给softmax使用
    #     y_predict = tf.matmul(x, weight) + bias

    y_predict = create_cnn_model(x)
    # 3、softmax回归以及交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):
        # labels:真实值 [None, 10]  one_hot
        # logits:全脸层的输出[None,10]
        # 返回每个样本的损失组成的列表
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
    # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # （2）收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)
    # 收集权重和偏置
    # tf.summary.histogram("weights", weight)
    # tf.summary.histogram("bias", bias)
    # 初始化变量op
    init_op = tf.global_variables_initializer()
    # （3）合并所有变量op
    merged = tf.summary.merge_all()
    # 创建模型保存和加载
    saver = tf.train.Saver()
    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # （1）创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)
        # # 加载模型
        # if os.path.exists("./tmp/modelckpt/checkpoint"):
        #     saver.restore(sess, "./tmp/modelckpt/fc_nn_model")
        if FLAGS.is_train == 1:
            # 循环步数去训练
            for i in range(3000):
                # 获取数据，实时提供
                # 每步提供50个样本训练
                mnist_x, mnist_y = mnist.train.next_batch(50)
                # 运行训练op
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                print("训练第%d步的准确率为：%f, 损失为：%f " % (i+1,
                                     sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                     sess.run(loss, feed_dict={x: mnist_x, y_true: mnist_y})
                                     )
                  )

                # 4）运行合变量op，写入事件文件当中
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                file_writer.add_summary(summary, i)
                # if i % 100 == 0:
                #     saver.save(sess, "./tmp/modelckpt/fc_nn_model")
        else:
            # 如果不是训练，我们就去进行预测测试集数据
            for i in range(100):
                # 每次拿一个样本预测
                mnist_x, mnist_y = mnist.test.next_batch(1)
                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                                                      i+1,
                                                      tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                                                      tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval()
                                                      )
                                                      )

    return None
if __name__ == "__main__":
    full_connection()