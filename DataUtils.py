# 把数据分成训练数据和测试数据
import math
import os
import numpy as np
import tensorflow as tf

abs_path = 'train'
DIR = os.path.abspath(abs_path)
# print(DIR)

# 数据打标签
label_list = []
image_list = []

class0 = []
init_label0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class1 = []
init_label1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

class2 = []

init_label2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

class3 = []

init_label3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

class4 = []

init_label4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

class5 = []

init_label5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

class6 = []

init_label6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

class7 = []

init_label7 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

class8 = []

init_label8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

class9 = []

init_label9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


# 拆分图片为训练数据和测试数据
def get_files(file_dir, ratio):
    """

    @param file_dir: 文件路径
    @param ratio: 拆分率
    """
    # 遍历10个训练文件夹
    path = os.path.join(file_dir, str(0))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label0)

    path = os.path.join(file_dir, str(1))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label1)

    path = os.path.join(file_dir, str(2))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label2)

    path = os.path.join(file_dir, str(3))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label3)

    path = os.path.join(file_dir, str(4))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label4)

    path = os.path.join(file_dir, str(5))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label5)

    path = os.path.join(file_dir, str(6))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label6)

    path = os.path.join(file_dir, str(7))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label7)

    path = os.path.join(file_dir, str(8))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label8)

    path = os.path.join(file_dir, str(9))
    for file in os.listdir(path):
        image_list.append(os.path.join(path, file))
        label_list.append(init_label9)

    #
    # print(len(label_list))
    # print(len(image_list))
    # shuffle打乱
    temp = np.array([image_list, label_list])  # 2*221
    temp = temp.transpose()  # 221 * 2
    np.random.shuffle(temp)  # 打乱行，图片与标签保持对应
    # print(temp.shape)
    # print(temp)
    # 打乱后的数据集
    all_image_list = list(temp[:,0])  # 第0列 image
    all_label_list = list(temp[:,1])  # 第1列 label
    # print(temp[:,0].shape)
    # print(temp[:,1].shape)
    # print(type(temp[:,0]))
    # print(type(temp[:,1]))
    # (221,)
    # (221,)

    # 拆分训练集和测试集
    n_sample = len(all_label_list)
    # 测试数量
    n_val = int(math.ceil(n_sample * ratio))
    # 训练数量
    n_train = n_sample - n_val  #  n_train = 112
    # 训练数据集
    train_img = all_image_list[:n_train]
    train_labels = all_label_list[:n_train]
    # print(len(train_img))
    # print(len(train_labels))  # tpye = list

    # 测试数据集
    val_images = all_image_list[n_train:]
    val_labels = all_label_list[n_train:]

    return train_img, train_labels, val_images, val_labels


"""
将图片转为 tensorFlow 能读取的张量
"""

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 数据转换 结果都是张量
    tf_image = tf.cast(image, tf.string)  # 将image数据转换为string类型
    tf_label = tf.cast(label, tf.int32)  # 将label数据转换为int类型
    # print(tf_image) # Tensor("Cast/x:0", shape=(112,), dtype=string)
    # print(tf_label) # Tensor("Cast_1/x:0", shape=(112, 10), dtype=int32)
    # 文件读写用多线程同时读写，用队列通信 样本和标签一一对应
    # 入文件名队列
    input_queue = tf.train.slice_input_producer([tf_image, tf_label])
    # 取队列标签 张量
    tf_label = input_queue[1]
    # print(tf_label) Tensor("input_producer/GatherV2_1:0", shape=(10,), dtype=int32)
    # 取队列图片
    image_contents = tf.read_file(input_queue[0])
    # print(image_contents) Tensor("ReadFile:0", shape=(), dtype=string)
    # 解码图像，解码为一个张量 3通道
    decode_image = tf.image.decode_jpeg(image_contents, channels=3)
    # 对图像的大小进行调整，调整大小为image_W,image_H
    re_image = tf.image.resize_image_with_crop_or_pad(decode_image, image_W, image_H)
    # 对图像进行标准化
    nor_image = tf.image.per_image_standardization(re_image)
    # print(nor_image)  # Tensor("per_image_standardization:0", shape=(28, 28, 3), dtype=float32)

    # 一个batch以后出队
    image_batch, label_batch = tf.train.batch([nor_image, tf_label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    # print(image_batch)     # Tensor("batch:0", shape=(10, 28, 28, 3), dtype=float32)
    # print(label_batch)     # Tensor("batch:1", shape=(10, 10), dtype=int32)


    return image_batch, label_batch  # 返回所处理得到的图像batch和标签batch
#
# train_img, train_labels, val_images, val_labels = get_files(DIR,0.2)
# get_batch(train_img, train_labels,28,28,10,256)
