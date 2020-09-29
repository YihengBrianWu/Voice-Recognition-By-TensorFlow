import tensorflow as tf

# 在上面已经创建了TFRecord文件，
# 为了可以在训练中读取TFRecord文件，创建reader.py程序用于读取训练数据，
# 如果读者已经修改了训练数据的长度，需要修改tf.io.FixedLenFeature中的值。

def _parse_data_function(example):
    # [可能需要修改参数】 设置的梅尔频谱的shape相乘的值

    data_feature_description = {
        'data': tf.io.FixedLenFeature([24064], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, data_feature_description)

def train_reader_tfrecord(data_path, num_epochs, batch_size):

    raw_dataset = tf.data.TFRecordDataset(data_path)
    train_dataset = raw_dataset.map(_parse_data_function)
    train_dataset = train_dataset.shuffle(buffer_size=1000) \
        .repeat(count=num_epochs) \
        .batch(batch_size=batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset

def test_reader_tfrecord(data_path, batch_size):

    raw_dataset = tf.data.TFRecordDataset(data_path)
    test_dataset = raw_dataset.map(_parse_data_function)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    return test_dataset
