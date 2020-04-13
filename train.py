import tensorflow as tf
import numpy as np
import DataUtils
import CNN
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2 as cv
import pretreatment as pt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

N_CLASSES = 10  # 10分类
capacity = 256  # 队列容量
BATCH_SIZE = 15
MAX_STEP = 500  # 最大训练步骤
IMG_W = 28  # 图片的宽度
IMG_H = 28  # 图片的高度
learning_rate = 0.0001  # 学习率
keep_prob = 0.7

"""
 定义开始训练的函数
"""


def training():
    """
    ##1.数据的处理
    """
    # 训练图片路径
    abs_path = 'train'
    train_dir = os.path.abspath(abs_path)

    # 输出log的位置
    abs_path = 'log'
    logs_train_dir = os.path.abspath(abs_path)
    # 模型输出
    abs_path = 'model'
    train_model_dir = os.path.abspath(abs_path)

    tra_list, tra_labels, val_list, val_labels = DataUtils.get_files(train_dir, 0.2)
    tra_list_batch, tra_label_batch = DataUtils.get_batch(tra_list, tra_labels, IMG_W, IMG_H, BATCH_SIZE,
                                                          capacity)  # 转成tensorflow 能读取的格式的数据
    # val_list_batch, val_label_batch = DataUtils.get_batch(val_list, val_labels, IMG_W, IMG_H, BATCH_SIZE,
    #                                                       capacity)

    #     print('Data Utils finished......')
    #     print(tra_list_batch, '\n',tra_label_batch)
    #     print(val_list_batch, '\n',val_label_batch)
    #     Tensor("batch:0", shape=(10, 28, 28, 3), dtype=float32)
    #     Tensor("batch:1", shape=(10, 10), dtype=int32)
    #     Tensor("batch_1:0", shape=(10, 28, 28, 3), dtype=float32)
    #     Tensor("batch_1:1", shape=(10, 10), dtype=int32)

    """
    ##2.网络的推理
    """
    # 进行前向训练，获得回归值
    prediction = CNN.CNN_moudle(tra_list_batch, BATCH_SIZE, N_CLASSES, keep_prob)
    # print(prediction)
    """
    ##3.定义交叉熵和 要使用的梯度下降的 优化器
    """
    # 二次loss
    cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tra_label_batch,
                                                                          logits=prediction))
    tf.summary.scalar('loss', cross_entroy)
    # Adam optimizer
    train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entroy)
    """
    ##4.定义后面要使用的变量
    """
    # tf.argmax(y,1) 求标签最大的值在第几个位置  axis=1 ,表示按照行向量比较
    # tf.argmax(prediction,1) 求预测最大的值在第几个位置
    # 一样的 返回 tool 向量，保存起来
    correct_prediction = tf.equal(tf.argmax(tra_label_batch, axis=1), tf.argmax(prediction, axis=1))
    # 求准确率 True = 1.0 False = 0
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # 合并所有的指标
    summary_op = tf.summary.merge_all()

    # 新建会话
    sess = tf.Session()

    # 将训练日志写入到logs_train_dir的文件夹内 sess.graph:结构图
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()  # 保存变量

    # 执行训练过程，初始化变量
    sess.run(tf.global_variables_initializer())

    # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    coord = tf.train.Coordinator()
    # 启动入队的线程，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    """
    进行训练：
    使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，
    会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;
    """

    try:
        for step in np.arange(MAX_STEP):  # 从0 到 500 次 循环

            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train, cross_entroy, accuracy])

            # 每2步打印一次损失值和准确率
            if step % 2 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.4f%%' % (step, tra_loss, tra_acc * 100.0))

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
    # 如果读取到文件队列末尾会抛出此异常
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()  # 使用coord.request_stop()来发出终止所有线程的命令

    coord.join(threads)  # coord.join(threads)把线程加入主线程，等待threads结束

    checkpoint_path = os.path.join(train_model_dir, 'model.ckpt')

    # saver.save(sess, checkpoint_path, global_step=step)

    saver.save(sess, checkpoint_path)
    sess.close()  # 关闭会话


def get_one_image_file(img_dir):
    image = Image.open(img_dir)
    plt.legend()
    plt.imshow(image)  # 显示图片
    image = image.resize([28, 28])
    image = np.array(image)
    return image


"""
进行ID图片的测试
"""


def evaluate_id_image():
    max_index = []
    filenames = os.listdir('test1')
    for file in filenames:
        print('test1/' + file)
        filenames.sort(key=lambda x: int(x[:-4]))
        image_array = get_one_image_file('test/' + file)

        with tf.Graph().as_default():
            BATCH_SIZE = 1  # 获取一张图片
            N_CLASSES = 10  # 10分类

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, 28, 28, 3])  # inference输入数据需要是4维数据，需要对image进行resize
            logit = CNN.CNN_moudle(image, BATCH_SIZE, N_CLASSES, keep_prob)
            logit = tf.nn.softmax(logit)  # inference的softmax层没有激活函数，这里增加激活函数

            # 因为只有一副图，数据量小，所以用placeholder
            x = tf.placeholder(tf.float32, shape=[28, 28, 3])

            #
            # 训练模型路径
            logs_train_dir = 'model/'

            saver = tf.train.Saver()

            with tf.Session() as sess:
                saver.restore(sess, str(logs_train_dir + "model.ckpt"))

                prediction = sess.run(logit, feed_dict={x: image_array})
                print(prediction)
                # 得到概率最大的索引
                index = np.argmax(prediction)
                max_index.append(index)
    print(max_index)


"""
主函数
"""


def pre_treatment():
    test_image_path = 'test01.jpg'
    # 读取彩色识别图片
    test_image = cv.imread(test_image_path, 1)
    test_image = test_image[0:-200, 1000::]
    test_image = cv.resize(test_image, (0, 0), fx=0.3, fy=0.3)
    cv.namedWindow("test_image", 0)  # flag = 0 ,默认窗口大小可以改变
    cv.resizeWindow("test_image", 800, 600)
    cv.imshow('test_image', test_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    pos = pt.pretreatment(test_image)
    # 根据确定的位置分割字符
    for m in range(len(pos)):
        """
        img图像 pt1矩形的一个顶点。pt2矩形对角线上的另一个顶点 
        color线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）。
        thickness组成矩形的线条的粗细程度。
        """
        cv.rectangle(test_image, (pos[m][0] - 3, pos[m][1] - 2),
                     (pos[m][2] + 3, pos[m][3] + 2),
                     (0, 0, 255), 1)
    cv.imwrite('rec_image.jpg', test_image)

    # 根据确定的位置分割字符
    print(len(pos))
    rec_image = cv.imread('test_image.png')
    for m in range(-18, 0, 1):
        data = rec_image[pos[m][1] - 1: pos[m][3] + 1, pos[m][0] - 1:pos[m][2] + 1]
        cv.imwrite(r'test1/{}.jpg'.format((m + 19)), data)





def main():
    try:
        pre_treatment()
        # training()
        # evaluate_id_image()
    except:
        print('请重新拍摄照片！')
    # training()
    # evaluate_id_image()
    # evaluate_ONE_image()


if __name__ == '__main__':
    main()
