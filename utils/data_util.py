from PIL import Image
import numpy as np
import tensorflow as tf
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("image_path", "./source_data/train/", "源图片的路径")
tf.flags.DEFINE_string("letter", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", "验证码字符的种类")

def create_dir_path(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print('成功创建路径:{}'.format(path))

def dealWithLabel(labelStr_list):
    """

    :param label_list: ['2a2s', '2w3e', '4e5r', '3e5r', '4r5g'....]
    :return:
    """
    letter = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

    # 构建字符索引 {0：'a', 1:'b'......}
    num2letter = dict(enumerate(list(letter)))
    # 键值对翻转 {'a':0, 'b':1......}
    letter2num = dict(zip(num2letter.values(),num2letter.keys()))
    print(letter2num)

    # 构建标签的列表
    array = []
    for lablex in labelStr_list:

        letter_list = []

        for letter in list(str(lablex)): # '2a2s'>>[2,a,2,s]
            letter2_num = letter2num[str(letter)] #a>>2
            letter_list.append(letter2_num)

        array.append(np.asarray(letter_list))

    array = np.asarray(array)

    return array

def numpy_normalization(data):
    """
    用numpy实现对数组的归一化
    :param data: 数组
    :return:
    """

    range = np.max(data) - np.min(data)
    a = (data-np.min(data))/range
    return np.asarray(a,dtype=np.float32)
def get_imagePath_label(path):
    """

    :param path: 图片的上级路径,如 './pic_source/train6000/'
    :return:
    image_path_list: ['./pic_source/train6000/2a2s.jpg', ''......]
    label_list: ["2a2s", ""......]
    """
    file_name_list = os.listdir(path=path) #结果>>>['2a2s.jpg', '', ''....]
    image_path_list = []
    labelStr_list = []
    for file_name in file_name_list:
        file_path = os.path.join(path,file_name)
        image_path_list.append(file_path)

        label_name = str(file_name).split(".")[0]
        labelStr_list.append(label_name)

    print("图像的目录:", image_path_list[:20])
    print("标签列表:", labelStr_list[:20])
    return image_path_list, labelStr_list


def get_image_label_list(image_path_list,
                         labelStr_list,
                         image_height=60,
                         image_weight=160):
    """
    将图片集合,标签集合转换成最终的tensor集合
    :param image_path_list: ['../pic_source/train6000/2a2s.jpg', ' '......]
    :param labelStr_list: ["2a2s", "2s2d"......]
    :return:
    """

    image_list = [] # 存放图片数据的数组
    # label_batch = [] # 存放图片标签的数组 注意存放的是代号 [2a2s]>>[3,24,3,56]
    for path in image_path_list:
        image = Image.open(path).convert('L')
        image = np.array(image.resize((image_height, image_weight)))
        image = numpy_normalization(image)    # 将数组归一化
        image = image.reshape((image_height, image_weight, 1),)  # 重塑形状
        image_list.append(image)

    label_list = dealWithLabel(labelStr_list)  #numpy类型
    image_list = np.asarray(image_list)
    print('image_batch:', type(image_list))
    print('label_batch:', type(label_list))

    return image_list, label_list


def generateCnn_image_label_batch(image_path='../source_data/valid',
                                  batch_size=50,
                                  label_num=4,
                                  label_depth=62,
                                  image_height = 60,
                                  image_weight = 160,
                                  is_make_label_onehot=True
                                  ):
    """

    :param image_path:
    :param batch_size:
    :param label_depth:
    :param image_height:
    :param image_weight:
    :return: image_batch (batch_size,image_height,image_weight,1)
             label_batch (batch_size,label_num=4,label_depth,1)

    """

    image_path_list, labelStr_list = get_imagePath_label(image_path)  #图像目录的列表,标签的列表
    image_list, label_list = get_image_label_list(image_path_list, labelStr_list,image_height,image_weight)  #图像的列表,标签转换为数字的列表

    if is_make_label_onehot:
        label_list = tf.one_hot(label_list, depth=label_depth, on_value=1.0, axis=-1) # (batch_size, 4)>> (batch_size, 4, 62)
        label_list = tf.reshape(label_list,shape=[-1,label_num*label_depth]) # (-1, 4, 62)>> (-1, 4*62)
    else:
        pass

    image_list = tf.cast(image_list, dtype=tf.float32)  #将图像list转换为tensor
    label_list = tf.cast(label_list, dtype=tf.int32)  #将标签list转换为tensor

    input_queue = tf.train.slice_input_producer([image_list, label_list], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=9000)


    print('image_batch的类型:', type(image_batch))
    print('label_batch的类型:', type(label_batch))
    return image_batch, label_batch

if __name__ == '__main__':


    image_batch, label_batch = generateCnn_image_label_batch('../source_data/valid', batch_size=5,is_make_label_onehot=False)

    with tf.Session() as sess:
        # 初始化参数
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 开启协调器,使用start_queue_runners 启动队列填充
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        #********************************************************************
        try:
            for i in range(1):  # 每一轮迭代
                for j in range(2):  # 每一个batch
                    print('第{}轮,第{}个批次--------'.format(i+1,j+1))
                    # 获取每一个batch中batch_size个样本和标签
                    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                    print(image_batch_v.shape, label_batch_v.shape,type(image_batch_v), label_batch_v[:, 0])
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)
        # ********************************************************************