import keras, os
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, InputSpec, Lambda, Average, CuDNNLSTM, Flatten, TimeDistributed, Dropout, concatenate, dot, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, UpSampling2D, UpSampling1D, AveragePooling1D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPool1D
from keras.models import load_model
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import pickle, os, re, numpy as np, gensim, time, sys
import pandas as pd, math, collections, random, tables
from sklearn.metrics import classification_report
import constants.deep_matching.constants as cnt
import shared_utilities as shutils
import utilities.deep_matching.utilities as utils
from keras_self_attention import SeqSelfAttention
import tensorflow as tf


def get_shared_model(wv_tensor, cv_tensor, is_train_tensor):
    with tf.name_scope("character_embedding"):
        n_layer = tf.reshape(cv_tensor, shape=[-1, cnt.MAX_WORDS*cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM], name="char-reshape-1")
        
    with tf.name_scope("char-conv-1"):
        w = tf.get_variable('W-char-conv', shape=[3, cnt.CHAR_VECTOR_DIM, 32], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('B-char-conv', shape=[32], initializer=tf.contrib.layers.xavier_initializer())

        n_layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(n_layer, w, stride=1, padding='SAME'), b))
        
    n_layer = tf.reshape(n_layer, shape=[-1, cnt.MAX_WORDS, cnt.MAX_CHARS, 32], name="char-reshape-2")

    with tf.name_scope("char-pool"):
        n_layer = tf.nn.avg_pool(n_layer, ksize=[1, 1, cnt.MAX_CHARS, 1], strides=[1, 1, cnt.MAX_CHARS, 1], padding='SAME')
        
    n_layer = tf.squeeze(n_layer)

    with tf.name_scope("concatanate"):
        n_layer = tf.concat([wv_tensor, n_layer], -1)

    with tf.name_scope("char-conv-2"):
        w1 = tf.get_variable('W1-char-conv', shape=[3, cnt.WORD_VECTOR_DIM + 32, 32], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('B1-char-conv', shape=[32], initializer=tf.contrib.layers.xavier_initializer())

        w2 = tf.get_variable('W2-char-conv', shape=[3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('B2-char-conv', shape=[32], initializer=tf.contrib.layers.xavier_initializer())

        n_layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(n_layer, w1, stride=1, padding='SAME'), b1))
        n_layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(n_layer, w2, stride=1, padding='SAME'), b2))
        n_layer = tf.layers.batch_normalization(n_layer, training=is_train_tensor)

        n_layer = tf.expand_dims(n_layer, -1)
        n_layer = tf.nn.max_pool(n_layer, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    n_layer = tf.squeeze(n_layer, [-1])
            
    return n_layer


def get_model(wv_tensor1, wv_tensor2, cv_tensor1, cv_tensor2, out_tensor, is_train_tensor):
    dim = int(cnt.MAX_WORDS/2)
    
    with tf.name_scope("shared-layer"):
        with tf.variable_scope("shared-arch", reuse=False):
            n_layer1 = get_shared_model(wv_tensor1, cv_tensor1, is_train_tensor)
            
        with tf.variable_scope("shared-arch", reuse=True):
            n_layer2 = get_shared_model(wv_tensor2, cv_tensor2, is_train_tensor)
        
    with tf.name_scope("merge"):
        n_layer = tf.matmul(n_layer1, n_layer2, transpose_a=False, transpose_b=True)
        
    n_layer = tf.expand_dims(n_layer, -1)
    
    with tf.name_scope("conv-merge"):
        w1 = tf.get_variable('W1-merge-conv', shape=[3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('B1-merge-conv', shape=[64], initializer=tf.contrib.layers.xavier_initializer())

        w2 = tf.get_variable('W2-merge-conv', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('B2-merge-conv', shape=[64], initializer=tf.contrib.layers.xavier_initializer())

        n_layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(n_layer, w1, strides=[1, 1, 1, 1], padding='SAME'), b1))
        n_layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(n_layer, w2, strides=[1, 1, 1, 1], padding='SAME'), b2))
        n_layer = tf.layers.batch_normalization(n_layer, training=is_train_tensor)

        n_layer = tf.nn.avg_pool(n_layer, ksize=[1, dim, dim, 1], strides=[1, dim, dim, 1], padding='SAME')

    n_layer = tf.squeeze(n_layer)
    
    with tf.name_scope("output"):
        w = tf.get_variable('W-out', shape=[64, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('B-out', shape=[1], initializer=tf.contrib.layers.xavier_initializer())

        output = tf.nn.xw_plus_b(n_layer, w, b, name='MODEL_PREDICTIONS')
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=out_tensor)
    predicted = tf.nn.sigmoid(output)
    correct_prediction = tf.cast(tf.equal(tf.round(predicted), out_tensor), tf.float32)

    return output, loss, correct_prediction


def get_loss_accuracy(wv_tensor1, wv_tensor2, cv_tensor1, cv_tensor2, out_tensor, is_train_tensor, num_gpus=cnt.USE_NUM_GPUS):
    if num_gpus <= 1:
        _, loss, correct_prediction = get_model(wv_tensor1, wv_tensor2, cv_tensor1, cv_tensor2, out_tensor, is_train_tensor)
            
        cost = tf.reduce_mean(loss)
        accuracy = tf.reduce_mean(correct_prediction)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
            
    else:
        wv_tensors1 = tf.split(wv_tensor1, num_gpus)
        cv_tensors1 = tf.split(cv_tensor1, num_gpus)
        
        wv_tensors2 = tf.split(wv_tensor2, num_gpus)
        cv_tensors2 = tf.split(cv_tensor2, num_gpus)
        
        out_tensors = tf.split(out_tensor, num_gpus)

        losses, correct_predictions = [], []
        
        for i in range(cnt.USE_NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    _, loss, correct_prediction = get_model(wv_tensors1[i], wv_tensors2[i], cv_tensors1[i], cv_tensors2[i], out_tensors[i], is_train_tensor)
                    losses.append(loss)
                    correct_predictions.append(correct_prediction)

        cost = tf.reduce_mean(tf.concat(losses, axis=0))
        accuracy = tf.reduce_mean(tf.concat(correct_predictions, axis=0))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.004).minimize(cost, colocate_gradients_with_ops=True)
    
    return optimizer, cost, accuracy


def get_predictions(wv_tensor1, wv_tensor2, cv_tensor1, cv_tensor2, out_tensor, is_train_tensor, num_gpus=cnt.USE_NUM_GPUS):
    if num_gpus <= 1:
        output, _, _ = get_model(wv_tensor1, wv_tensor2, cv_tensor1, cv_tensor2, out_tensor, is_train_tensor)
            
    else:
        wv_tensors1 = tf.split(wv_tensor1, num_gpus)
        cv_tensors1 = tf.split(cv_tensor1, num_gpus)
        
        wv_tensors2 = tf.split(wv_tensor2, num_gpus)
        cv_tensors2 = tf.split(cv_tensor2, num_gpus)
        
        out_tensors = tf.split(out_tensor, num_gpus)

        outputs = []
        for i in range(cnt.USE_NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    output, _, _ = get_model(wv_tensors1[i], wv_tensors2[i], cv_tensors1[i], cv_tensors2[i], out_tensors[i], is_train_tensor)
                    outputs.append(output)

        output = tf.concat(outputs, axis=0)
    
    return output


def process_batches(num_batches, batch_iter, wv_tensor1, wv_tensor2, cv_tensor1, cv_tensor2, out_tensor, is_train_tensor, optimizer, cost, accuracy, sess, mode='train'):
    c_loss, c_acc, curr_n = 0, 0, 0

    for j in range(num_batches):
        x_data, y_data = next(batch_iter)
        
        word_data_1, word_data_2, char_data_1, char_data_2 = x_data
        n = word_data_1.shape[0]

        if n % cnt.USE_NUM_GPUS != 0:
            m = int(n/cnt.USE_NUM_GPUS)
            m *= cnt.USE_NUM_GPUS
            
            word_data_1 = word_data_1[:m]
            word_data_2 = word_data_2[:m]
            char_data_1 = char_data_1[:m]
            char_data_2 = char_data_2[:m]
            
            y_data = y_data[:m]

        if mode == 'train':
            opt, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={wv_tensor1: word_data_1, wv_tensor2: word_data_2, cv_tensor1: char_data_1, cv_tensor2: char_data_2, out_tensor: y_data, is_train_tensor: True})
            
        else:
            loss, acc = sess.run([cost, accuracy], feed_dict={wv_tensor1: word_data_1, wv_tensor2: word_data_2, cv_tensor1: char_data_1, cv_tensor2: char_data_2, out_tensor: y_data, is_train_tensor: False})
        
        curr_n += n
        c_loss += loss*n
        c_acc += acc*n
        
        if j % 100 == 0:
            print(c_loss/curr_n, c_acc/curr_n)


    c_loss /= curr_n
    c_acc /= curr_n
    
    return c_loss, c_acc, sess


class DeepMatchingNetwork:
    def __init__(self, data_generator=None, num_train=None, num_test=None):
        self.data_generator = data_generator
        self.num_test = num_test
        self.num_train = num_train
        
        self.wv1 = tf.placeholder(tf.float32, [None, cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM], name="WV_1")
        self.wv2 = tf.placeholder(tf.float32, [None, cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM], name="WV_2")
        self.cv1 = tf.placeholder(tf.float32, [None, cnt.MAX_WORDS, cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM], name="CV_1")
        self.cv2 = tf.placeholder(tf.float32, [None, cnt.MAX_WORDS, cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM], name="CV_2")
        
        self.output = tf.placeholder(tf.float32, [None, 1], name="output")
        self.training = tf.placeholder(tf.bool, name="batch_norm_training_bool")
        
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        
        self.sess = tf.Session(config=session_conf)
        
        self.output_op = None
    
    def fit(self):
        with self.sess.as_default():
            optimizer, cost, accuracy = get_loss_accuracy(self.wv1, self.wv2, self.cv1, self.cv2, self.output, self.training)
            
            train_summary_writer = tf.summary.FileWriter(cnt.TF_TRAIN_SUMMARY_PATH, self.sess.graph)
            test_summary_writer = tf.summary.FileWriter(cnt.TF_TEST_SUMMARY_PATH, self.sess.graph)
            
            saver = tf.train.Saver()

            self.sess.run(tf.global_variables_initializer())
            
            steps_per_epoch_train=shutils.get_num_batches(self.num_train, cnt.BATCH_SIZE)
            steps_per_epoch_test=shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE)
            
            min_test_loss = float("Inf")

            for i in range(cnt.NUM_EPOCHS):
                train_iter = self.data_generator(self.num_train, 'train')
                test_iter = self.data_generator(self.num_test, 'test')
                
                train_c_loss, train_c_acc, self.sess = process_batches(steps_per_epoch_train, train_iter, self.wv1, self.wv2, self.cv1, self.cv2, self.output, self.training, optimizer, cost, accuracy, self.sess, mode='train')
                
                train_summary = tf.Summary()
                train_summary.value.add(tag="Accuracy", simple_value=train_c_acc)
                train_summary.value.add(tag="Loss", simple_value=train_c_loss)
                    
                train_summary_writer.add_summary(train_summary, i)
                
                test_c_loss, test_c_acc, self.sess = process_batches(steps_per_epoch_test, test_iter, self.wv1, self.wv2, self.cv1, self.cv2, self.output, self.training, optimizer, cost, accuracy, self.sess, mode='test')
                
                test_summary = tf.Summary()
                test_summary.value.add(tag="Accuracy", simple_value=test_c_acc)
                test_summary.value.add(tag="Loss", simple_value=test_c_loss)
                
                test_summary_writer.add_summary(test_summary, i)
                
                if test_c_loss < min_test_loss:
                    min_test_loss = test_c_loss
                    saver.save(self.sess, cnt.MODEL_PATH)

                print("Iter " + str(i) + ", Training Loss= " + "{:.6f}".format(train_c_loss) + ", Training Accuracy= " + "{:.5f}".format(train_c_acc))
                print("Iter " + str(i) + ", Validation Loss= " + "{:.6f}".format(test_c_loss) + ", Validation Accuracy= " + "{:.5f}".format(test_c_acc))
                print()

            train_summary_writer.close()
            test_summary_writer.close()
            
    
    def predict(self, test_data, threshold=0.5, return_probability=False):
        if self.output_op is None:
            self.output_op = get_predictions(self.wv1, self.wv2, self.cv1, self.cv2, self.output, self.training)
            saver = tf.train.Saver()
            saver.restore(self.sess, cnt.MODEL_PATH)
        
        word_data_1, word_data_2, char_data_1, char_data_2 = test_data
        
        n = word_data_1.shape[0]
        
        if n % cnt.USE_NUM_GPUS != 0:
            m = int(n/cnt.USE_NUM_GPUS) + 1
            m *= cnt.USE_NUM_GPUS
            
            dummy_data_w1 = np.zeros((m-n, word_data_1.shape[1], word_data_1.shape[2]))
            dummy_data_w2 = np.zeros((m-n, word_data_2.shape[1], word_data_2.shape[2]))
            
            dummy_data_c1 = np.zeros((m-n, char_data_1.shape[1], char_data_1.shape[2], char_data_1.shape[3]))
            dummy_data_c2 = np.zeros((m-n, char_data_2.shape[1], char_data_2.shape[2], char_data_2.shape[3]))
            
            word_data_1 = np.vstack((word_data_1, dummy_data_w1))
            word_data_2 = np.vstack((word_data_2, dummy_data_w2))
            char_data_1 = np.vstack((char_data_1, dummy_data_c1))
            char_data_2 = np.vstack((char_data_2, dummy_data_c2))

        preds = self.sess.run([self.output_op], feed_dict={self.wv1: word_data_1, self.wv2: word_data_2, self.cv1: char_data_1, self.cv2: char_data_2, self.training: False})[0]
        
        preds = preds[:n]
        preds = 1.0/(1.0+np.exp(-preds))
        
        outs, probs = [], []
        
        for i in range(preds.shape[0]):
            g, h = np.copy(preds[i]), np.copy(preds[i])
            g[h > threshold] = 1
            g[h <= threshold] = 0
            outs.append(g.tolist())
            probs.append(h[h > threshold].tolist())
        
        outs = [[int(x) for x in y] for y in outs]
            
        if return_probability:
            return outs, probs
        
        return outs
    
    def scoring(self):
        test_labels, pred_labels, total_batches = [], [], shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE)
        
        num_batches = 0
        for batch_data, batch_labels in self.data_generator(self.num_test, 'test'):
            test_labels += batch_labels.tolist()
            pred_labels += self.predict(batch_data).tolist()
            num_batches += 1
            if num_batches == total_batches:
                break
        
        print(classification_report(test_labels, pred_labels))
        
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, cnt.MODEL_PATH)
        
    def load(self):
        self.output_op = get_predictions(self.wv1, self.wv2, self.cv1, self.cv2, self.output, self.training)
        saver = tf.train.Saver()
        saver.restore(self.sess, cnt.MODEL_PATH)
