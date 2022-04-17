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
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
                         for unit, act_fn in zip(unit_list, act_fn_list)])
    
class LSTM_VAE(object):
    def __init__(self,dataset_name, columns, z_dim, time_steps, outlier_fraction):
        self.outlier_fraction = outlier_fraction
        self.data_source = Data_Hanlder(dataset_name, columns, time_steps)
        self.n_hidden = 32
        self.batch_size = 16
        self.learning_rate = 0.0001
        self.train_iters = 100000
        
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
            self.X = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.input_dim],name='input_X')
        
        with tf.variable_scope('encoder'):
            with tf.variable_scope('lat_mu'):
                mu_fw_lstm_cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.z_dim)
                mu_bw_lstm_cells = tf.nn.rnn_cell.BasicLSTMCell(self.z_dim)
                init_state = mu_fw_lstm_cells.zero_state(self.batch_size,dtype=tf.float32)
                init_random = tf.random_normal_initializer(mean = 0.0,stddev= 1.0)
                # (mu_fw_outputs, mu_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                #                                         mu_fw_lstm_cells,
                #                                         mu_bw_lstm_cells,
                #                                         self.X, dtype=tf.float32)
                self.outputs, self.cell_states = tf.nn.dynamic_rnn(mu_fw_lstm_cells, inputs=self.X, initial_state = init_state,dtype=tf.float32)
                # self.outputs = tf.add(mu_fw_outputs, mu_bw_outputs)
                print('mu_outputs:', self.outputs)

                # encode_reshaped = tf.reshape(outputs,[])
                # print('reshaped:', encode_reshaped)
                self.mu = tf.layers.dense(self.outputs, 3)
                # self.sigma = tf.layers.dense(self.outputs, 3, activation= tf.nn.softplus)
                self.sigma = tf.layers.dense(self.outputs, 3)
                print('sigma:', self.sigma)
                self.sample_Z = self.mu + self.sigma * tf.random_normal(
                                                        tf.shape(self.mu),
                                                        0,1, dtype=tf.float32)


        with tf.variable_scope('decoder'):

            recons_lstm_cells = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden,activation=tf.tanh)
            print('recons_lstm_cells:', self.sample_Z)
            self.recons_X, _ = tf.nn.dynamic_rnn(recons_lstm_cells, self.sample_Z, dtype=tf.float32)
            print('recons_X:', self.recons_X)
            # decode_reshaped = tf.layers.flatten(self.recons_X)
            self.recons_mu = tf.layers.dense(self.recons_X, self.input_dim)
            self.recons_sigma = tf.layers.dense(self.recons_X, self.input_dim, activation=tf.nn.softplus)




        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1, tf.keras.backend.ndim(self.X))
            # self.recons_loss = 0.5 * tf.reduce_mean(tf.losses.mean_squared_error(self.X, self.recons_mu) +
            #                                    tf.log(tf.reduce_sum(self.X)))
            self.recons_loss = tf.losses.mean_squared_error(self.X, self.recons_mu)
            print('recons_loss:', self.recons_loss.shape)
            self.kl_loss = - 0.5 * tf.reduce_mean(1 + tf.log(tf.square(self.sigma+ 1e-8)) - tf.square(self.mu+ 1e-8) - tf.square(self.sigma+ 1e-8))
            print('kl_loss:', self.kl_loss.shape)
            self.opt_loss = self.recons_loss + self.kl_loss
            # self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X), reduction_indices=reduce_dims)
            self.anomaly_score = tf.reduce_mean(((self.X - self.recons_mu)**2)/2*(self.recons_sigma**2) +
                                                tf.log(self.recons_sigma))
        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)
            
    def cal_anomaly(self):
        self.anomaly_score = - tf.reduce_mean(tf.log((self.X - self.recons_mu)/self.recons_sigma))
    def train(self):
        saver = tf.train.Saver()

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state('./model/')
            # print("ckpt:",ckpt.model_checkpoint_path)
            try :
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print("加载模型!")
            except Exception as e:
                print("无模型")
            for i in range(self.train_iters):
                this_X = self.data_source.fetch_data(batch_size=self.batch_size)
                loss,Z, mu, output, mse_loss, kl, recon, sigma = self.sess.run(
                    [self.uion_train_op,self.sample_Z, self.mu, self.outputs, self.opt_loss, self.kl_loss, self.recons_loss,self.sigma],
                    feed_dict={
                        self.X: this_X
                    })
                # print("sigma:", sigma)
                # print("mu:", mu)
                # print("output:", output)
                # if (math.isnan(loss[0])):
                #     print('Z:{}'.format(Z))
                #     print('output:{}'.format(output))
                #     break
                # this_X,label = read_batch('./dataset/data0.csv',self.batch_size)
                # print("this_x:", this_X)
                # if (np.isnan(loss).sum()>0):
                #     print('loss,nan:',loss)
                #     break
                # print("X isnan:", np.isnan(this_X).sum())
                # print("anomaly_score:", anomaly_score)
                if i %200 == 0:


                        # Z = self.sess.run([self.sample_Z],feed_dict={self.X :this_X})
                        # print('mse:{},log:{},sum:{}'.format(mse,log,sum))

                    # print('cell_states:{}'.format(cell_states))

                    # print('sigma.shape:{}'.format(sigma.shape))
                    # print('output:{}'.format(output))
                    #     print("mu:",mu)
                    print('recon_loss:{},kl_loss:{}'.format(recon,kl))
                    # print('mse:{}'.format(mse))
                    #
                        # print('output:{}'.format(output))
                    # print('z:{}'.format(Z))
                        # print('X.shape:{}'.format(this_X.shape))
                    # print('X:{}'.format(this_X))

                    print('round {}: with loss: {}'.format(i, mse_loss))
                # self.sess.run(self.uion_train_op, feed_dict={
                #     self.X: this_X
                # })
                if i > 0 and i % 2000 ==0:
                    saveName = "model/VAE-LSTM_" + str(i)
                    saver.save(self.sess, saveName)

            # self._arange_score(self.data_source.train)

       
    def judge(self,test):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model/')
        # print("ckpt:",ckpt.model_checkpoint_path)
        try:
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("加载模型!")
        except Exception as e:
            print("无模型")

        anomaly_score = self.sess.run(self.anomaly_score, feed_dict={
                                    self.X: test
                                    })
        io.savemat('异常分数.mat', {'data': anomaly_score})

        return anomaly_score


    def plot_confusion_matrix(self):
        print(self.data_source.test.shape)
        predict_label = self.judge(self.data_source.test)
        # self.data_source.plot_confusion_matrix(self.data_source.test_label,predict_label,['Abnormal','Normal'],'LSTM_VAE Confusion-Matrix')


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
            print("测试集:",self.data_source.test.shape)
            np.save('dataset/recon.npy', data)
        print("完成！")

def main():

    lstm_vae = LSTM_VAE('dataset/data0.csv', ['v0','v1','X','Y','Z'], z_dim=16, time_steps=64, outlier_fraction=0.01)
    lstm_vae.train()
    # lstm_vae.plot_confusion_matrix()
    # lstm_vae.get_rec_data()
if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    main()

