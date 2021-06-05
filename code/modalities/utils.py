import tensorflow as tf


def int64_list_feature(int64_list) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def float_list_feature(float_list) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_list))


def bytes_list_feature(bytes_list) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))
