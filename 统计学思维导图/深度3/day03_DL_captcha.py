import tensorflow as tf
import pandas as pd
import glob
import numpy as np


# 流程分析：
# 1）解析CSV文件, 建立文件名和标签值对应表格
def parse_csv():

    # 读取csv文件
    csv_data = pd.read_csv("./GenPics/labels.csv", names=["file_num", "chars"], index_col="file_num")

    # 建立一个空列表
    labels = []
    for label in csv_data["chars"]:
        tmp = []
        for letter in label:
            #         print(ord(letter) - ord("A"))
            tmp.append(ord(letter) - ord("A"))
        labels.append(tmp)

    csv_data["labels"] = labels

    return csv_data

# 2）读取图片数据
def read_image():

    # 生成文件名列表
    file_list = glob.glob("./GenPics/*.jpg")
    # print("文件名列表：\n", file_list)

    # 1、构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2、读取与解码
    # 读取
    reader = tf.WholeFileReader()
    filename, value = reader.read(file_queue)

    # 解码
    image = tf.image.decode_jpeg(value)
    # 固定形状
    image.set_shape([20, 80, 3])
    # print("image:\n", image)

    # 3、批处理
    filename_batch, image_batch = tf.train.batch([filename, image], batch_size=100, num_threads=2, capacity=200)

    return filename_batch, image_batch

# 3）将标签值的字母转为0~25的数字
def filename2label(filenames, csv_data):
    # print("filenames:\n", filenames)
    labels = []
    for filename in filenames:
        file_num = "".join(list(filter(str.isdigit, str(filename))))
        # print("file_num:\n", type(file_num))
        labels.append(csv_data.loc[int(file_num), "labels"])

    # print("labels:\n", labels)

    return np.array(labels)
# 4）建立卷积神经网络模型
def create_variable(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.01))

def create_cnn_model(x):

    with tf.variable_scope("conv1"):
        # 卷积层

        conv1_weights = create_variable([5, 5, 3, 32])
        # conv1_bias = tf.Variable(tf.random_normal(shape=[32]))
        conv1_bias = create_variable([32])
        conv1_tensor = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 形状变化 [None, 20, 80, 3] --> [None, 20, 80, 32]

        # 激活层
        relu1_tensor = tf.nn.relu(conv1_tensor)

        # 池化层
        pool1_tensor = tf.nn.max_pool(relu1_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 形状变化 [None, 20, 80, 32] --> [None, 10, 40, 32]

    with tf.variable_scope("conv2"):
        # 卷积层

        conv2_weights = create_variable([5, 5, 32, 64])
        conv2_bias = create_variable([64])

        conv2_tensor = tf.nn.conv2d(pool1_tensor, conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_tensor = tf.nn.relu(conv2_tensor)

        # 池化层
        pool2_tensor = tf.nn.max_pool(relu2_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 形状改变 [None, 10, 40, 32] --> [None, 5, 20, 64]

    with tf.variable_scope("fc"):
        # [None, 5, 20, 64] --> [None, 5*20*64]
        # [None, 5*20*64] * [5*20*64, 4*26] = [None, 4*26]
        fc_x = tf.reshape(pool2_tensor, [-1, 5*20*64])

        fc_weights = create_variable([5*20*64, 4*26])
        fc_bias = create_variable([4*26])
        #
        # # y_predict (None, 10)
        y_predict = tf.matmul(fc_x, fc_weights) + fc_bias

    return y_predict

if __name__ == "__main__":
    csv_data = parse_csv()

    filename_batch, image_batch = read_image()


    # 准备数据，定义占位符
    x = tf.placeholder(tf.float32, [None, 20, 80, 3])
    y_true = tf.placeholder(tf.float32, [None, 4*26])
    # 构建模型
    y_predict = create_cnn_model(x)

    # 5）计算sigmoid交叉熵损失
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 6）优化
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 7）计算准确率
    equal_list = tf.reduce_all(tf.equal(
    tf.argmax(tf.reshape(y_true, [-1, 4, 26]), axis=-1),
    tf.argmax(tf.reshape(y_predict, [-1, 4, 26]), axis=-1)), axis=-1)
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 变量初始化
    init = tf.global_variables_initializer()

    # 8）模型训练
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):

            filenames, images = sess.run([filename_batch, image_batch])

            labels = filename2label(filenames, csv_data)

            labels_onehot = tf.reshape(tf.one_hot(labels, 26), [-1, 4*26]).eval()

            _, loss_value, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={x: images, y_true: labels_onehot})

            print("第%d次训练的结果，损失为%f, 准确率为%f" % (i+1, loss_value, accuracy_value))
        # print("labels_onehot:\n", labels_onehot)
        # print("one_hot:\n", sess.run(labels_onehot))

        # print("filenames:\n", filenames, type(filenames))
        # print("images:\n", images)
        # 回收线程
        coord.request_stop()
        coord.join(threads)
