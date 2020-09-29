# 导入依赖
import os
import random

import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 获取浮点数组

def _float_feature(value):

    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

# 获取整型数组
def _int64_feature(value):

    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

# 把数据添加到tf.train.Example里
def data_example(data, label):

    feature = {
        'data': _float_feature(data),
        'label': _int64_feature(label),
    }

    return tf.train.Example(features = tf.train.Features(feature = feature))

# 开始创建tfrecord数据
# 把语音数据转换成梅尔频谱，使用librosa库的API：librosa.feature.melspectrogram()
# 使用librosa.effects.split减掉低音部分，减少数据集的噪音
# 默认时长为2.04秒 当然可以随意修改
# 如果语音长度比较长的，程序会随机裁剪20次，以达到数据增强的效果
def create_data_tfrecord(data_list_path, save_path):

    with open(data_list_path, 'r') as f:
        data = f.readlines()

    with tf.io.TFRecordWriter(save_path) as writer:
        for d in tqdm(data):
            try:
                path, label = d.replace('\n', '').split('\t')
                # 装载wav文件
                wav, sr = librosa.load(path, sr = 16000)
                # 切掉音量不足的部分，阈值为20分贝
                intervals = librosa.effects.split(wav, top_db=20)
                wav_output = []
                # [可能需要修改参数] 音频长度 16000 * 秒数
                wav_len = int(16000 * 3)

                for sliced in intervals:
                    wav_output.extend(wav[sliced[0]:sliced[1]])

                # 逻辑：判断是否过长，是的话先剪裁
                # 如果没有过长，则随机打乱这段已经读取的数据
                # 使得数据更加具有随机性，使数据的可使用性更高
                for i in range(20):
                    if len(wav_output) >= wav_len:
                        l = len(wav_output) - wav_len
                        r = random.randint(0, l)
                        wav_output = wav_output[r:wav_len + r]
                    else:
                        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))

                    wav_output = np.array(wav_output)

                    # 转成梅尔频谱
                    ps = librosa.feature.melspectrogram(y = wav_output, sr = sr, hop_length = 256).reshape(-1).tolist()
                    # [可能需要修改参数] 梅尔频谱shape ，librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).shape

                    if len(ps) != 24064: continue
                    tf_example = data_example(ps, int(label))
                    writer.write(tf_example.SerializeToString())
                    if len(wav_output) <= wav_len:
                        break
            except Exception as e:
                print(e)

# 生成数据列表
def get_data_list(audio_path, list_path):

    # 打开audio路径，读取全部的audio
    files = os.listdir(audio_path)

    label_dict = {}
    i = 0
    for file in files:
        if '.wav' not in file:
            continue
        if file[10:15] in label_dict:
            continue
        label_dict[file[10:15]] = i
        i += 1

    # 做两个路径，一个是训练集一个测试集
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    # 分离测试和训练数据，每99个训练数据1个测试数据
    total = 0

    # for循环处理数据
    for file in files:
        # .wav测试，如果不是wav格式直接跳过


        if '.wav' not in file:
            continue

        sound_path = os.path.join(audio_path, file)

        label = label_dict[file[10:15]]

        # 分割数据集
        if total % 100 == 0:
            f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), label))
        else:
            f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), label))

        total += 1

    # 关闭写入流
    f_test.close()
    f_train.close()

if __name__ == '__main__':
    get_data_list('dataset/ST-CMDS-20170001_1-OS', 'dataset')
    create_data_tfrecord('dataset/train_list.txt', 'dataset/train.tfrecord')
    create_data_tfrecord('dataset/test_list.txt', 'dataset/test.tfrecord')
