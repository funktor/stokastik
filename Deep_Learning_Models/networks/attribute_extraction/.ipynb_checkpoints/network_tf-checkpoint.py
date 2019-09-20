# from ctypes import *
# lib = CDLL("/usr/local/cuda-10.0/lib64/libcudnn.so.7")

import keras, os, importlib
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

from keras.models import Model, Input
from keras.layers import Dense, Lambda, Flatten, Dropout, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.models import load_model
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import pickle, os, re, numpy as np, gensim, time, sys
import pandas as pd, math, collections, random, tables
from sklearn.metrics import classification_report, precision_recall_fscore_support
import constants.attribute_extraction.constants as cnt
import shared_utilities as shutils
utils = importlib.import_module('utilities.attribute_extraction.' + os.environ['ATTRIBUTE'] + '.utilities')
import tensorflow as tf


def get_model(img_tensor, txt_tensor, out_tensor, num_classes, vocab_size, is_train_tensor):
    with tf.name_scope("img_model"):
        curr_num_channels, img_h, img_w = 3, cnt.IMAGE_SIZE, cnt.IMAGE_SIZE
        n_layer1 = img_tensor
        
        if cnt.USE_TRANSFER_LEARNING_IMAGE:
            base_model = tf.keras.applications.VGG16(input_tensor=n_layer1, include_top=False, weights='imagenet')

            for layer in base_model.layers[:13]:
                layer.trainable =  False

            n_layer1 = base_model.output
            n_layer1 = tf.keras.layers.GlobalAveragePooling2D()(n_layer1)
            
        else:
            for i, num_filter in enumerate(cnt.IMAGE_NUM_FILTERS):
                with tf.name_scope("img-conv-maxpool-%s" % num_filter):
                    w1 = tf.get_variable('W1-image-%s' % i, shape=[3, 3, curr_num_channels, num_filter], initializer=tf.contrib.layers.xavier_initializer())
                    b1 = tf.get_variable('B1-image-%s' % i, shape=[num_filter], initializer=tf.contrib.layers.xavier_initializer())

                    curr_num_channels = num_filter

                    w2 = tf.get_variable('W2-image-%s' % i, shape=[3, 3, curr_num_channels, num_filter], initializer=tf.contrib.layers.xavier_initializer())
                    b2 = tf.get_variable('B2-image-%s' % i, shape=[num_filter], initializer=tf.contrib.layers.xavier_initializer())

                    curr_num_channels = num_filter

                    n_layer1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(n_layer1, w1, strides=[1, 1, 1, 1], padding='SAME'), b1))
                    n_layer1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(n_layer1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2))
                    n_layer1 = tf.layers.batch_normalization(n_layer1, training=is_train_tensor)

                    if i == len(cnt.IMAGE_NUM_FILTERS)-1:
                        n_layer1 = tf.nn.avg_pool(n_layer1, ksize=[1, img_h, img_w, 1], strides=[1, img_h, img_w, 1], padding='SAME')
                        img_h, img_w = 1, 1

                    else:
                        n_layer1 = tf.nn.max_pool(n_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        img_h, img_w = img_h/2, img_w/2


    with tf.name_scope("txt_model"):
        curr_len, dim = cnt.WORD_VECTOR_DIM, cnt.MAX_WORDS
        n_layer2 = txt_tensor
        
        n_layer2 = tf.keras.layers.Embedding(vocab_size, cnt.WORD_VECTOR_DIM, input_length=cnt.MAX_WORDS)(n_layer2)

        for i, num_filter in enumerate(cnt.TEXT_NUM_FILTERS):
            with tf.name_scope("txt-conv-maxpool-%s" % num_filter):
                w1 = tf.get_variable('W1-text-%s' % i, shape=[3, curr_len, num_filter], initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.get_variable('B1-text-%s' % i, shape=[num_filter], initializer=tf.contrib.layers.xavier_initializer())

                curr_len = num_filter

                w2 = tf.get_variable('W2-text-%s' % i, shape=[3, curr_len, num_filter], initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable('B2-text-%s' % i, shape=[num_filter], initializer=tf.contrib.layers.xavier_initializer())

                curr_len = num_filter

                n_layer2 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv1d(n_layer2, w1, stride=1, padding='SAME'), b1))
                n_layer2 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv1d(n_layer2, w2, stride=1, padding='SAME'), b2))
                n_layer2 = tf.layers.batch_normalization(n_layer2, training=is_train_tensor)

                n_layer2 = tf.expand_dims(n_layer2, -1)

                if i == len(cnt.TEXT_NUM_FILTERS)-1:
                    n_layer2 = tf.nn.avg_pool(n_layer2, ksize=[1, dim, 1, 1], strides=[1, dim, 1, 1], padding='SAME')
                    dim = 1

                else:
                    n_layer2 = tf.nn.max_pool(n_layer2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
                    dim = dim/2

                n_layer2 = tf.squeeze(n_layer2, [-1])

    with tf.name_scope("concatentation"):
        n_layer1 = tf.squeeze(n_layer1)
        n_layer2 = tf.squeeze(n_layer2)

        n_layer = tf.concat([n_layer1, n_layer2], -1)

    model_out_shape = cnt.IMAGE_NUM_FILTERS[-1] + cnt.TEXT_NUM_FILTERS[-1]

    with tf.name_scope("output"):
        w = tf.get_variable('W-out', shape=[model_out_shape, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('B-out', shape=[num_classes], initializer=tf.contrib.layers.xavier_initializer())

        output = tf.nn.xw_plus_b(n_layer, w, b, name='MODEL_PREDICTIONS')
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=out_tensor)
    correct_prediction = tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(out_tensor, 1)), tf.float32)

    return output, loss, correct_prediction


def get_loss_accuracy(img_tensor, txt_tensor, out_tensor, num_classes, vocab_size, is_train_tensor, num_gpus=cnt.USE_NUM_GPUS):
    img_tensors = tf.split(img_tensor, num_gpus)
    txt_tensors = tf.split(txt_tensor, num_gpus)
    out_tensors = tf.split(out_tensor, num_gpus)

    losses, correct_predictions = [], []
    for i in range(cnt.USE_NUM_GPUS):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                _, loss, correct_prediction = get_model(img_tensors[i], txt_tensors[i], out_tensors[i], num_classes, vocab_size, is_train_tensor)
                losses.append(loss)
                correct_predictions.append(correct_prediction)

    cost = tf.reduce_mean(tf.concat(losses, axis=0))
    accuracy = tf.reduce_mean(tf.concat(correct_predictions, axis=0))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, colocate_gradients_with_ops=True)
    
    return optimizer, cost, accuracy


def get_predictions(img_tensor, txt_tensor, out_tensor, num_classes, vocab_size, is_train_tensor, num_gpus=cnt.USE_NUM_GPUS):
    img_tensors = tf.split(img_tensor, num_gpus)
    txt_tensors = tf.split(txt_tensor, num_gpus)
    out_tensors = tf.split(out_tensor, num_gpus)

    outputs = []
    for i in range(cnt.USE_NUM_GPUS):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                output, _, _ = get_model(img_tensors[i], txt_tensors[i], out_tensors[i], num_classes, vocab_size, is_train_tensor)
                outputs.append(output)

    output = tf.concat(outputs, axis=0)
    
    return output


def process_batches(num_batches, batch_iter, img_tensor, txt_tensor, out_tensor, is_train_tensor, optimizer, cost, accuracy, sess, mode='train'):
    c_loss, c_acc, curr_n = 0, 0, 0

    for j in range(num_batches):
        x_data, y_data = next(batch_iter)
        
        img_data = x_data[0]
        txt_data = x_data[1]

        if img_data.shape[0] % cnt.USE_NUM_GPUS != 0:
            m = int(img_data.shape[0]/cnt.USE_NUM_GPUS)
            m *= cnt.USE_NUM_GPUS
            
            img_data = img_data[:m]
            txt_data = txt_data[:m]
            y_data = y_data[:m]

        if mode == 'train':
            opt, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={img_tensor: img_data, txt_tensor: txt_data, out_tensor: y_data, is_train_tensor: True})
            
        else:
            loss, acc = sess.run([cost, accuracy], feed_dict={img_tensor: img_data, txt_tensor: txt_data, out_tensor: y_data, is_train_tensor: False})

        curr_n += img_data.shape[0]
        c_loss += loss*img_data.shape[0]
        c_acc += acc*img_data.shape[0]

    c_loss /= curr_n
    c_acc /= curr_n
    
    return c_loss, c_acc, sess


class AttributeExtractionNetwork:
    def __init__(self, data_generator=None, num_train=None, num_test=None, num_classes=None, vocab_size=None):
        self.data_generator = data_generator
        self.num_test = num_test
        self.num_train = num_train
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        
        self.input_img = tf.placeholder(tf.float32, [None, cnt.IMAGE_SIZE, cnt.IMAGE_SIZE, 3], name="input_img")
        self.input_txt = tf.placeholder(tf.float32, [None, cnt.MAX_WORDS], name="input_txt")
        self.output = tf.placeholder(tf.float32, [None, self.num_classes], name="output")
        self.training = tf.placeholder(tf.bool, name="batch_norm_training_bool")
        
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        
        self.sess = tf.Session(config=session_conf)
        
        self.output_op = None
            
    
    def fit(self):
        with self.sess.as_default():
            optimizer, cost, accuracy = get_loss_accuracy(self.input_img, self.input_txt, self.output, self.num_classes, self.vocab_size, self.training)
            
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
                
                train_c_loss, train_c_acc, self.sess = process_batches(steps_per_epoch_train, train_iter, self.input_img, self.input_txt, self.output, self.training, optimizer, cost, accuracy, self.sess, mode='train')
                
                train_summary = tf.Summary()
                train_summary.value.add(tag="Accuracy", simple_value=train_c_acc)
                train_summary.value.add(tag="Loss", simple_value=train_c_loss)
                    
                train_summary_writer.add_summary(train_summary, i)
                
                test_c_loss, test_c_acc, self.sess = process_batches(steps_per_epoch_test, test_iter, self.input_img, self.input_txt, self.output, self.training, optimizer, cost, accuracy, self.sess, mode='test')
                
                test_summary = tf.Summary()
                test_summary.value.add(tag="Accuracy", simple_value=test_c_acc)
                test_summary.value.add(tag="Loss", simple_value=test_c_loss)
                
                test_summary_writer.add_summary(test_summary, i)
                
                if cnt.SAVE_BEST_LOSS_MODEL:
                    if test_c_loss < min_test_loss:
                        min_test_loss = test_c_loss
                        saver.save(self.sess, cnt.MODEL_PATH)
                else:
                    saver.save(self.sess, cnt.MODEL_PATH)

                print("Iter " + str(i) + ", Training Loss= " + "{:.6f}".format(train_c_loss) + ", Training Accuracy= " + "{:.5f}".format(train_c_acc))
                print("Iter " + str(i) + ", Validation Loss= " + "{:.6f}".format(test_c_loss) + ", Validation Accuracy= " + "{:.5f}".format(test_c_acc))
                print()

            train_summary_writer.close()
            test_summary_writer.close()
        
    
    def predict(self, test_data, threshold=0.5, return_probability=False):
        if self.output_op is None:
            self.output_op = get_predictions(self.input_img, self.input_txt, self.output, self.num_classes, self.vocab_size, self.training)
            saver = tf.train.Saver()
            saver.restore(self.sess, cnt.MODEL_PATH)
        
        img_data = test_data[0]
        txt_data = test_data[1]
        
        n = img_data.shape[0]
        
        if n % cnt.USE_NUM_GPUS != 0:
            m = int(n/cnt.USE_NUM_GPUS) + 1
            m *= cnt.USE_NUM_GPUS
            
            dummy_data_img = np.zeros((m-n, img_data.shape[1], img_data.shape[2], img_data.shape[3]))
            dummy_data_txt = np.zeros((m-n, txt_data.shape[1]))
            
            img_data = np.vstack((img_data, dummy_data_img))
            txt_data = np.vstack((txt_data, dummy_data_txt))

        preds = self.sess.run([self.output_op], feed_dict={self.input_img: img_data, self.input_txt: txt_data, self.training: False})[0]
        preds = preds[:n]
        
        if cnt.IS_MULTILABEL:
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0
            
        else:
            h = np.argmax(preds, axis=1)
            preds[:,:] = 0
            preds[range(preds.shape[0]), h] = 1
        
        return preds.tolist()
    
    def scoring(self):
        test_labels, pred_labels, total_batches = [], [], shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE)
        encoder = shutils.load_data_pkl(cnt.ENCODER_PATH)
        
        num_batches = 0
        
        for batch_data, batch_labels in self.data_generator(self.num_test, 'test'):
            test_labels += batch_labels.tolist()
            predictions = self.predict(batch_data)
            pred_labels += predictions
            num_batches += 1
            
            if num_batches == total_batches:
                break
        
        t_labels = encoder.inverse_transform(np.array(test_labels))
        p_labels = encoder.inverse_transform(np.array(pred_labels))
        
        print(classification_report(t_labels, p_labels, target_names=encoder.classes_))
        
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, cnt.MODEL_PATH)
        
    def load(self):
        if cnt.LOAD_SAVED_GRAPH:
            saver = tf.train.import_meta_graph(cnt.MODEL_PATH + '.meta')
            saver.restore(self.sess, cnt.MODEL_PATH)
            
            graph = tf.get_default_graph()
            
            self.input_img = graph.get_tensor_by_name("input_img_1:0")
            self.input_txt = graph.get_tensor_by_name("input_txt_1:0")
            self.training = graph.get_tensor_by_name("batch_norm_training_bool_1:0")
            
        else:
            output = get_predictions(self.input_img, self.input_txt, self.output, self.num_classes, self.vocab_size, self.training)
            
            saver = tf.train.Saver()
            saver.restore(self.sess, cnt.MODEL_PATH)