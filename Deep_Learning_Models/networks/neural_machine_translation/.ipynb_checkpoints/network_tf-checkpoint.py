from __future__ import absolute_import, division, print_function, unicode_literals
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from ctypes import *
lib = CDLL("/usr/local/cuda-10.0/lib64/libcudnn.so.7")
import unicodedata
import re
import numpy as np
import os
import io
import time
import constants.neural_machine_translation.constants as cnt
import utilities.neural_machine_translation.utilities as utils
import shared_utilities as shutils
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, 
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_initializer='glorot_uniform')
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.enc_units))
    
    
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def call(self, x, hidden, enc_output, is_training):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.batch_norm(output, training=is_training)
        
        x = self.fc(output)

        return x, state, attention_weights
    
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(src, trg, trg_lang, encoder, decoder, optimizer, batch_size, type='train'):
    loss = 0
    enc_hidden = encoder.initialize_hidden_state(batch_size)
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(src, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([trg_lang.word_index['<start>']] * batch_size, 1)

        for t in range(1, trg.shape[1]):
            if type == 'train':
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, is_training=True)
            else:
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, is_training=False)
                
            loss += loss_function(trg[:, t], predictions)
            dec_input = tf.expand_dims(trg[:, t], 1)

    batch_loss = (loss / int(trg.shape[1]))
    
    if type == 'train':
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

class NMTNetwork:
    def __init__(self, data_generator=None, num_train=None, num_valid=None, src_lang=None, trg_lang=None, max_length_src=None, max_length_trg=None):
        self.data_generator = data_generator
        self.num_valid = num_valid
        self.num_train = num_train
        self.trg_lang = trg_lang
        self.src_lang = src_lang
        self.max_length_src = max_length_src
        self.max_length_trg = max_length_trg
        
        self.encoder = Encoder(len(self.src_lang.word_index)+1, cnt.ENCODER_EMB_DIM, cnt.ENCODER_UNITS, cnt.BATCH_SIZE)
        self.decoder = Decoder(len(self.trg_lang.word_index)+1, cnt.DECODER_EMB_DIM, cnt.DECODER_UNITS, cnt.BATCH_SIZE)
        
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
    
    def predict(self, src):
        self.checkpoint.restore(tf.train.latest_checkpoint(cnt.PERSISTENCE_PATH))
        
        src = utils.preprocess_sentence(src)
        src = [self.src_lang.word_index[i] for i in src.split(' ') if i in self.src_lang.word_index]
        src = tf.keras.preprocessing.sequence.pad_sequences([src], maxlen=self.max_length_src, padding='post')
        
        src = tf.convert_to_tensor(src)
        result = ''

        hidden = [tf.zeros((1, cnt.ENCODER_UNITS))]
        enc_out, enc_hidden = self.encoder(src, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.trg_lang.word_index['<start>']], 0)

        for t in range(self.max_length_trg):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_out, is_training=False)
            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.trg_lang.index_word[predicted_id] + ' '

            if self.trg_lang.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

        return result
        
    def fit(self):
        curr_best_validation_loss = float("Inf")
        
        for epoch in range(cnt.NUM_EPOCHS):
            train_iter = self.data_generator(self.num_train, 'train')
            valid_iter = self.data_generator(self.num_valid, 'valid')
            
            steps_per_epoch_train=shutils.get_num_batches(self.num_train, cnt.BATCH_SIZE)
            steps_per_epoch_valid=shutils.get_num_batches(self.num_valid, cnt.BATCH_SIZE)

            total_loss = 0
            
            for batch in range(steps_per_epoch_train):
                src, trg = next(train_iter)
                batch_size = src.shape[0]
                batch_loss = train_step(src, trg, self.trg_lang, self.encoder, self.decoder, self.optimizer, batch_size, type='train')
                total_loss += batch_loss
                
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.8f}'.format(epoch + 1, batch, batch_loss.numpy()))
                    
            print('Epoch {} Mean Training Loss {:.8f}'.format(epoch + 1, total_loss/self.num_train))
            
            for batch in range(steps_per_epoch_valid):
                src, trg = next(valid_iter)
                batch_size = src.shape[0]
                batch_loss = train_step(src, trg, self.trg_lang, self.encoder, self.decoder, self.optimizer, batch_size, type='valid')
                total_loss += batch_loss
                
            validation_loss = total_loss/self.num_valid
            
            if validation_loss < curr_best_validation_loss:
                curr_best_validation_loss = validation_loss
                self.checkpoint.save(file_prefix = cnt.MODEL_PATH)
            
            print('Epoch {} Mean Validation Loss {:.8f}'.format(epoch + 1, validation_loss))
            
     
