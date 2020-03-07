import tensorflow as tf
import numpy as np

# def conv2d(name, input, w, conv_stride=1, use_bn=True,is_train=True):
#     # 卷积 + 批归一化+ Relu激活函数
#     with tf.name_scope(name=name), tf.variable_scope(name):
#         w_c = tf.Variable(tf.truncated_normal(w, stddev=0.1, mean=0.0),name='w_c')
#         # b_c = tf.Variable(tf.truncated_normal([b], stddev=0.1), name='b_c')
#         if use_bn:
#             net = tf.nn.conv2d(input, w_c, strides=[1, conv_stride, conv_stride, 1], padding='SAME', name='conv')
#             net = tf.layers.batch_normalization(net, training=is_train)
#         else:
#             net = tf.nn.conv2d(input, w_c, strides=[1, 1, 1, 1], padding='SAME', name='conv')
#     net = tf.nn.relu6(net)
#     return net

def batch_norm(x, train=True, name='bn'):
    return tf.layers.batch_normalization(x, training=train, name=name)


def conv2d(input, output_dims, k_h, k_w, d_h, d_w, stddev=0.1, name='conv2d', bias=False):
    """
    构建一个基础的卷积过程
    :param input:
    :param output_dims:
    :param k_h:
    :param k_w:
    :param d_h:
    :param d_w:
    :param stddev:
    :param name:
    :param bias:
    :return:
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_h, k_w, input.get_shape()[-1], output_dims],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding='SAME')

        if bias:
            biases = tf.get_variable('bias', shape=[output_dims], initializer=tf.zeros_initializer())
            conv = tf.nn.bias_add(conv, biases)
    return conv



def conv2d_block(input, output_dim, k, s, name):
    """
    实现的是 卷积 + 批归一化 +  激活函数
    :param input:
    :param output_dim:
    :param k:
    :param s:
    :param is_train:
    :param name:
    :return:
    """
    # with tf.name_scope(name):
    #     with tf.variable_scope(name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, output_dim, k, k, s, s, name='conv2d')
        # net = batch_norm(net, train=is_train, name='bn')
        net = tf.nn.relu6(net)
    return net


def conv_1_1(input, output_dim, name, bias=False):
    """
    实现逐点卷积
    :param input:
    :param output_dim:
    :param name:
    :param bias:
    :return:
    """
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, output_dim, 1, 1, 1, 1, name=name, bias=bias)
    return net


def pointwise_block(input, output_dim, is_train, name, bias=False):
    """
    实现逐点卷积模块
    :param input:
    :param output_dim:
    :param is_train:
    :param name:
    :param bias:
    :return:
    """
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv_1_1(input, output_dim, bias=bias, name='pwb')
        net = batch_norm(net, train=is_train, name='bn')
        net = tf.nn.relu6(net)
    return net


def res_bottleneck_block(input, expansion_rate, output_dim, stride, is_train, name,
                         bias=False, shortcut=True):
    """
    构建Mobile_Net-v2的瓶颈模块
    """
    with tf.name_scope(name), tf.variable_scope(name):
        # 一、1*1 逐点卷积
        bottleneck_dim = round(expansion_rate * input.get_shape().as_list()[-1])
        net = conv_1_1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = tf.nn.relu6(net)

        # 二、dw 深度智能卷积
        net = dpethwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = tf.nn.relu6(net)

        # todo :加Senet block.
        # se0 = tf.layers.average_pooling2d(net, pool_size=net.get_shape()[1], strides=1)
        # # output shape = [-1, 1, 1, input_channels]
        # filters_num = se0.get_shape()[-1]
        # se1 = tf.layers.conv2d(
        #     se0, filters=int(filters_num/8), kernel_size=1, strides=1, activation=tf.nn.relu
        # )
        # se2 = tf.layers.conv2d(
        #     se1, filters=filters_num, kernel_size=1, strides=1, activation=tf.nn.sigmoid
        # )
        # # output shape = [-1, 1, 1, input_channels]
        # se2 = tf.reshape(se2, shape=[-1, filters_num])
        # se_out = net * se2

        # 三、pw & Linear
        net = conv_1_1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # 决定是否连接shortcut 只在stride=1的时候做shortcut
        if shortcut and stride == 1:
            in_dim = int(input.get_shape()[-1])
            if in_dim != output_dim:
                ins = conv_1_1(input, output_dim, name='ex_dim')
                net = ins + net
            else:
                net = input + net
        return net


def dpethwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=[1, 1, 1, 1],
                   padding='SAME', stddev=0.1, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        input_channels = input.get_shape()[-1]
        w = tf.get_variable('w', shape=[k_h, k_w, input_channels, channel_multiplier],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        # 深度智能卷积
        conv = tf.nn.depthwise_conv2d(input, filter=w, strides=strides, padding=padding)

        if bias:
            biases = tf.get_variable('bias', shape=[input_channels * channel_multiplier],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
    return conv


def global_avg(input,name):
    with tf.variable_scope('global_avg'):
        net = tf.layers.average_pooling2d(input, input.get_shape().as_list()[1:-1], 1, name=name)
    return net

def flatten(input):
    return tf.contrib.layers.flatten(input)


if __name__ == '__main__':
    a_array = np.array([[1, 2, 3], [4, 5, 6]])
    b_list = [[1, 2, 3], [3, 4, 5]]
    c_tensor = tf.constant([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]]])

    print(c_tensor.get_shape())
    print(c_tensor.get_shape().as_list()[0:2])


    # with tf.Session() as sess:
    #     print(sess.run(tf.shape(a_array)))
    #     print(sess.run(tf.shape(b_list)))
    #     print(sess.run(tf.shape(c_tensor)))
