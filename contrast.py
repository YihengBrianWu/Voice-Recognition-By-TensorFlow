# 下面开始实现声纹对比，创建infer_contrast.py程序，在加载模型时，
# 不要直接加载整个模型，而是加载模型的最后分类层的上一层，
# 这样就可以获取到语音的特征数据。
# 实现对比

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model('models/resnet.h5', compile = False)
intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)

# 读取音频数据
def load_data(data_path):

    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []

    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    assert len(wav_output) >= 8000, "有效音频小于0.5s"

    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps

def infer(audio_path):

    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature

if __name__ == '__main__':

    # 要预测的两个人的音频文件
    # person1 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00001A0001.wav'
    person1 = 'testdata/arctic_a0002.wav'
    person2 = 'testdata/arctic_a0001.wav'
    feature1 = infer(person1)[0]
    feature2 = infer(person2)[0]

    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > 0.8:
        print("%s 和 %s 为同一个人，相似度为：%f" % (person1, person2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (person1, person2, dist))
