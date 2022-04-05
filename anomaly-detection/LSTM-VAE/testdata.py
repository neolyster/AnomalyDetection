import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename_queue = tf.train.string_input_producer(["./dataset/data0.csv"])

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
# key返回的是读取文件和行数信息;value是按行读取到的原始字符串，送到decoder解析

record_defaults = [[1.0], [1.0], [1.0], [1.0],[1.0]]
# 这里的数据类型和文件数据类型一致，必须是list形式
data = tf.decode_csv(value, record_defaults=record_defaults)

features=data[1:5]
labels = data[0]
features_batch,labels_batch = tf.train.batch([features,labels], batch_size=16, capacity=3*16)
print(features_batch)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(600):
#         example, label = sess.run([features_batch,labels_batch])
#         print(example,label,i)
#
#     coord.request_stop()
#     coord.join(threads)
