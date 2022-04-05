import numpy as np
from scipy import io
import tensorflow as tf
# print(tf.test.is_gpu_available())
loadData = np.load('./dataset/train.npy')
# testData = np.load('./dataset/test.npy')
# reconData = np.load('./dataset/recon.npy')
# io.savemat('重建数据.mat',{'data': reconData})
io.savemat('训练集.mat',{'data': loadData})
# io.savemat('测试集.mat',{'data': testData})

# print('测试集:',testData.shape)
print('训练集:',loadData.shape)
# print('重建集:',reconData.shape)
