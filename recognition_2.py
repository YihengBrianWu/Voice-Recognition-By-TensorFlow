# recognition_2脚本，不依赖音频输入，将语音保存再进行匹配
# 导入全部所需要的依赖
import os
import wave
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
from tensorflow.keras.models import Model

# 池化层的名字，方便导入模型
layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model('models/resnet.h5')
intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)

person_feature = []
person_name = []

# 读取音频数据
def load_data(data_path):

    wav, sr = librosa.load(data_path, sr = 16000)
    intervals = librosa.effects.split(wav, top_db = 20)

    wav_output = []

    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])

    if len(wav_output) < 8000:
        raise Exception("有效音频小于0.5")

    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y = wav_output, sr = sr, hop_length = 256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]

    return ps

# 封装方法
def infer(audio_path):

    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature

# 加载要识别的语音库
def load_audio_db(audio_db_path):

    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)[0]
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)

# 识别函数
def recognition(path):

    name = ''
    pro = 0
    feature = infer(path)[0]

    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]

    return name, pro

# 主函数
if __name__ == '__main__':

    load_audio_db('audio_db')

    name, p = recognition('recognition/...')

    if p > 0.7:
        print("识别说话的为：%s，相似度为：%f" % (name, p))
    else:
        print("音频库没有该用户的语音")