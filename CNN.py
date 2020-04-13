# CNN 卷积神经网络模型
"""
input --> conv1 --> pool1 --> relu激活 --> conv2 --> pool2  --> relu激活 --> dropout --> output --> softmax

"""
import tensorflow as tf


# 一共140张  28 * 28 channels = 3 batch = 10

def CNN_moudle(input,batch_size,classes,keep_prob):


    """参数概要"""
    def variable_summaries(var):
        # 平均数
        with tf.name_scope('summaries'):
            mean=tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            # 标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('stddev', tf.reduce_min(var))
                tf.summary.histogram('var_histogram',var)

    """conv layer 1"""
    with tf.variable_scope('layer1'):
        re_input = tf.reshape(input, [batch_size, 28, 28, 3])  # [batch,h,w,channels]
        # 权重初始化 [h,w,in_channels,n_kernel]
        w1 = tf.get_variable(name='w1',dtype=tf.float32,
                         initializer=tf.truncated_normal(shape=[3,3,3,16],stddev=0.1,dtype=tf.float32))
        variable_summaries(w1)

        b1 = tf.get_variable(name='b1',dtype=tf.float32,initializer=tf.constant(0.1,shape=[16]))
        variable_summaries(b1)

        # strides=[batch,h,w,channels]
        conv1=tf.nn.conv2d(re_input,w1,strides=[1,1,1,1],padding='SAME')
        conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1,b1))

        """pooling layer 1"""
        # ksize=[batch,h,w,channels],strides=[batch,h,w,channels]
        pool1 = tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    """conv layer 2"""
    with tf.variable_scope('layer2'):
        # 权重初始化 [h,w,in_channels,n_kernel]
        w2 = tf.get_variable(name='w2', dtype=tf.float32,
                             initializer=tf.truncated_normal(shape=[3, 3, 16, 64], stddev=0.1, dtype=tf.float32))
        variable_summaries(w2)
        # 偏置值初始化
        b2 = tf.get_variable(name='b2', dtype=tf.float32, initializer=tf.constant(0.1, shape=[64]))
        variable_summaries(b2)
        # strides=[batch,h,w,channels]
        conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME')
        conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2, b2))

        """pooling layer 2"""
        # ksize=[batch,h,w,channels],strides=[batch,h,w,channels]
        pool2 = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """Full connection layer 1"""
    with tf.variable_scope('Full_connection_layer1'):
        w_fc1 = tf.get_variable(name='w_fc1', dtype=tf.float32,
                         initializer=tf.truncated_normal(shape=[7 * 7 * 64, 200], stddev=0.1, dtype=tf.float32))
        variable_summaries( w_fc1)
        b_fc1 = tf.get_variable(name='b_fc1',dtype=tf.float32,initializer=tf.constant(0.1,shape=[200]))
        h_pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # 把pool2输出扁平化为1d
        variable_summaries(b_fc1)
        # 全连接层输出
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    """dropout layer 1"""
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """Full connection layer 2"""
    with tf.variable_scope('Full_connection_layer2'):
        w_fc2 = tf.get_variable(name='w_fc2', dtype=tf.float32,
                            initializer=tf.truncated_normal(shape=[200, classes], stddev=0.1, dtype=tf.float32))
        variable_summaries(w_fc2)
        b_fc2 = tf.get_variable(name='b_fc2',dtype=tf.float32,initializer=tf.constant(0.1,shape=[classes]))
        variable_summaries(b_fc2)
        # 全连接层输出
        prediction = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

    return prediction # [1* 10]













