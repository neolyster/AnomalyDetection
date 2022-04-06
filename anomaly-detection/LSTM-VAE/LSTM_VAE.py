# -*- coding: utf-8 -*-
"""
One simple Implementation of LSTM_VAE based algorithm for Anomaly Detection in Multivariate Time Series;

Author: Schindler Liang

Reference:
    https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
    https://github.com/twairball/keras_lstm_vae
    https://arxiv.org/pdf/1711.00614.pdf    
"""
import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import MultiRNNCell, LSTMCell
from utils import Data_Hanlder
import os
from scipy import io

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def read_batch(filenames,batchsize):
    filename_queue = tf.train.string_input_producer([filenames])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    # key返回的是读取文件和行数信息;value是按行读取到的原始字符串，送到decoder解析

    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0]]
    # 这里的数据类型和文件数据类型一致，必须是list形式
    data = tf.decode_csv(value, record_defaults=record_defaults)

    features = data[1:5]
    labels = data[-1]
    features_batch, labels_batch = tf.train.batch([features, labels], batch_size=batchsize, capacity=3 * 16)
    return features_batch,labels_batch

def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)


def _LSTMCells(unit_list,act_fn_list):
    return MultiRNNCell([LSTMCell(unit,                         
                         activation=act_fn) 
                         for unit,act_fn in zip(unit_list,act_fn_list )])
    
class LSTM_VAE(object):
    def __init__(self,dataset_name,columns,z_dim,time_steps,outlier_fraction):
        self.outlier_fraction = outlier_fraction
        self.data_source = Data_Hanlder(dataset_name,columns,time_steps)
        self.n_hidden = 32
        self.batch_size = 32
        self.learning_rate = 100
        self.train_iters = 1000000
        
        self.input_dim = len(columns)
        self.z_dim = z_dim
        self.time_steps = time_steps
        self.pointer = 0 
        self.anomaly_score = 0
        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        
    def _build_network(self):
        with tf.variable_scope('ph'):
            self.X = tf.placeholder(tf.float32,shape=[None,self.time_steps,self.input_dim],name='input_X')
        
        with tf.variable_scope('encoder'):
            with tf.variable_scope('lat_mu'):
                mu_fw_lstm_cells = _LSTMCells([self.z_dim],[tf.nn.softplus])
                mu_bw_lstm_cells = _LSTMCells([self.z_dim],[tf.nn.softplus])

                (mu_fw_outputs,mu_bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                        mu_fw_lstm_cells,
                                                        mu_bw_lstm_cells, 
                                                        self.X, dtype=tf.float32)
                mu_outputs = tf.add(mu_fw_outputs,mu_bw_outputs)
                encode_reshaped = tf.keras.backend.flatten(mu_outputs)
                self.mu = tf.layers.dense(encode_reshaped, 3)
                self.sigma = tf.layers.dense(encode_reshaped, 3)

                self.sample_Z = self.mu + tf.log(self.sigma) * tf.random_normal(
                                                        tf.shape(self.mu),
                                                        0,1,dtype=tf.float32)


        with tf.variable_scope('decoder'):
            recons_lstm_cells = _LSTMCells(self.n_hidden, tf.tanh)
            self.recons_X,_ = tf.nn.dynamic_rnn(recons_lstm_cells, self.sample_Z, dtype=tf.float32)
            decode_reshaped = tf.keras.backend.flatten(self.recons_X)
            self.recons_mu= tf.layers.dense(decode_reshaped, self.time_steps)
            self.recons_sigma = tf.layers.dense(decode_reshaped, self.time_steps,activation=tf.nn.softplus)




        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1,tf.keras.backend.ndim(self.X))
            recons_loss = 0.5 * (tf.losses.mean_squared_error(self.X, self.mu) + tf.log(self.X))
            kl_loss = - 0.5 * tf.reduce_mean(1 + self.sigma - tf.square(self.mu) - tf.exp(self.sigma))
            self.opt_loss = recons_loss + kl_loss
            self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X), reduction_indices=reduce_dims)
            self.anomaly_score = tf.reduce_mean(((self.X - self.recons_mu)**2)/2*(self.recons_sigma**2) +
                                                tf.log(self.recons_sigma))
        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)
            
    def cal_anomaly(self):
        self.anomaly_score = - tf.reduce_mean(tf.log((self.X - self.recons_mu)/self.recons_sigma))
    def train(self):
        saver = tf.train.Saver()
        this_X = self.data_source.fetch_data(batch_size=self.batch_size)
        with self.sess.as_default():
            for i in range(self.train_iters):
                # this_X,label = read_batch('./dataset/data0.csv',self.batch_size)

                self.sess.run(self.uion_train_op, feed_dict={
                        self.X: this_X
                        })
                print("anomaly_score:", anomaly_score)
                if i % 200 ==0:
                    mse_loss = self.sess.run([self.opt_loss],feed_dict={
                        self.X: this_X
                        })
                    print('round {}: with loss: {}'.format(i,mse_loss))
                    # Z = self.sess.run([self.sample_Z],feed_dict={self.X :this_X})
                    # print('z:',np.shape(Z))

                if i % 500 ==0:
                    saveName = "model/VAE-LSTM_" + str(i)
                    saver.save(self.sess, "model/VAE-LSTM")

            # self._arange_score(self.data_source.train)

       
    def judge(self,test):
        anomaly_score = self.sess.run(self.anomaly_score, feed_dict={
                                    self.X: test
                                    })
        io.savemat('异常分数.mat', {'data': anomaly_score})

        return anomaly_score


    def plot_confusion_matrix(self):

        predict_label = self.judge(self.data_source.test)
        self.data_source.plot_confusion_matrix(self.data_source.test_label,predict_label,['Abnormal','Normal'],'LSTM_VAE Confusion-Matrix')


    def get_rec_data(self):
        self.data_source.test = np.load('./dataset/test.npy')
        self.data_source.test_label = np.load('./dataset/test_label.npy')
        ckpt = tf.train.latest_checkpoint('./model')  # 找到存储变量值的位置
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt)  # 加载到当前环境中
        print('加载模型!')
        with self.sess.as_default():
            data = self.sess.run(self.recons_X,feed_dict={
                self.X:self.data_source.test
            })
            np.save('dataset/recon.npy',data)
        print("完成！")

def main():

    lstm_vae = LSTM_VAE('dataset/data0.csv',['v0','v1','X','Y','Z'],z_dim=16,time_steps=64,outlier_fraction=0.01)
    lstm_vae.train()
    lstm_vae.plot_confusion_matrix()
    lstm_vae.get_rec_data()
if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    main()

