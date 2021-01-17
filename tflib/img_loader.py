# -*- coding: utf-8 -*-


from glob import glob
import numpy as np
import os
import pdb
import scipy.misc
import numpy as np
import time

trainset_path     = "/home/blues/workspace/kuangyan/f-AnoGAN/image/normal-training-t3"
trainset_val_path = "/home/blues/workspace/kuangyan/f-AnoGAN/image/normal-validation"
test_normal_path  = "/home/blues/workspace/kuangyan/f-AnoGAN/image/normal-testing"
test_anom_path    = "/home/blues/workspace/kuangyan/f-AnoGAN/image/anom-testing"

def get_files(data_set):
        if data_set == 'train_normal':
            # glob()方法返回所有匹配的文件路径列表（list）
            return glob(os.path.join(trainset_path, "*.png"))
        if data_set == 'valid_normal':
            return glob(os.path.join(trainset_val_path, "*.png"))
        elif data_set == 'test_normal':
            return glob(os.path.join(test_normal_path, "*.png"))
        elif data_set == 'test_anom':
            return glob(os.path.join(test_anom_path, "*.png"))

def get_nr_training_samples(batch_size):
    files = glob(os.path.join(trainset_path, "*.png"))
    total_nr_samples = len(files)
    nr_training_samples = total_nr_samples - np.mod(total_nr_samples, batch_size) # 取余，且符号与除数相同

    return nr_training_samples

def get_nr_samples(data_set, batch_size):
    files = get_files(data_set)
    total_nr_samples = len(files)
    nr_samples = total_nr_samples - np.mod(total_nr_samples, batch_size)

    return nr_samples

def get_nr_test_samples(batch_size):
    return (get_nr_samples('test_normal', batch_size),
             get_nr_samples('test_anom', batch_size))

def make_generator(data_set, batch_size):
    epoch_count = [1]

    def get_epoch():
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')

        files = get_files(data_set)
        assert(len(files) > 0) # assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达式为假

        # epoch_count[0]是伪随机数产生器的种子，只要该种子（seed）相同，产生的随机数序列就是相同
        random_state = np.random.RandomState(epoch_count[0]) # epoch_count[0]=1
        random_state.shuffle(files) # 将一个序列中的元素，随机打乱。
        epoch_count[0] += 1

        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for n, f in enumerate(files): # n是索引，f是文件元素file
            image = scipy.misc.imread(f, mode='L') # 将图片读取出来为array类型，即numpy类型，'L' 是指(8-bit pixels, black and white)
            if np.random.rand() >= 0.5: # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
                image = image[:, ::-1] # 数组反转
            images[n % batch_size] = np.expand_dims(image, 0) # 用于扩充数组维度，不懂的话:https://blog.csdn.net/qq_35860352/article/details/80463111
            
            if n > 0 and n % batch_size == 0:
                yield (images,) # yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，
    return get_epoch


def make_ad_generator(data_set, batch_size):

    def get_epoch():
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')

        files = get_files(data_set)
        nr_files = len(files)
        assert(nr_files > 0)

        for n, f in enumerate(files):
            image = scipy.misc.imread(f, mode='L')
            images[n % batch_size] = np.expand_dims(image, 0)

            if (n+1) % batch_size == 0:
                yield (images,)
            elif (n+1) == nr_files:
                final_btchsz = (n % batch_size)+1
                yield (images[:final_btchsz],)
    return get_epoch


def load(batch_size, run_type):
    if 'train' in run_type:
        return (
            make_generator('train_normal', batch_size),
            make_generator('valid_normal', batch_size)
        )
    elif run_type == 'anomaly_score':
        return (
            make_ad_generator('test_normal', batch_size),
            make_ad_generator('test_anom', batch_size)
        )


if __name__ == '__main__':
    print("here1")
    train_gen, valid_gen = load(16, 'encoder_train')
    t0 = time.time()
    print t0

    for n, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0, 0, 0, 0])
        if n == 1000:
            break
        t0 = time.time()

    # lena = mpimg.imread('/home/kim/kuangyan/work/f-AnoGAN/image/normal-training/000.png')# 读取和代码处于同一目录下的 lena.png
    # # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    # lena.shape # (64, 64, 1)
    # plt.imshow(lena)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()