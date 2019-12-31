from __future__ import absolute_import, division, print_function, unicode_literals
from ctypes import *
lib = CDLL("/usr/local/cuda-10.0/lib64/libcudnn.so.7")
import unicodedata
import re
import numpy as np
import os, math
import io
import time
import tensorflow as tf
import Utilities as utils

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
        
    def call(self, x, hidden, enc_output, is_training):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))
        
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
def train_step(src, trg, trg_lang, encoder, decoder, optimizer, batch_size, start_token):
    loss = 0.0
    enc_hidden = encoder.initialize_hidden_state(batch_size)
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(src, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([trg_lang.word_index[start_token]] * batch_size, 1)

        for t in range(1, trg.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, is_training=True)
                
            loss += loss_function(trg[:, t], predictions)
            dec_input = tf.expand_dims(trg[:, t], 1)

    batch_loss = (loss / int(trg.shape[1]))
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


class NMTNetwork:
    def __init__(self, data_generator, src_tensor, trg_tensor, src_tokenizer, trg_tokenizer, max_length_src, max_length_trg, start_token, end_token):
        
        self.data_generator = data_generator
        self.src_tensor = src_tensor
        self.trg_tensor = trg_tensor
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_length_src = max_length_src
        self.max_length_trg = max_length_trg
        self.batch_size = 64
        self.num_epochs = 200
        self.encoder_embed_dim = 1024
        self.decoder_embed_dim = 1024
        self.encoder_units = 1024
        self.decoder_units = 1024
        self.start_token, self.end_token = start_token, end_token
    
    def init_network(self):
        encoder = Encoder(len(self.src_tokenizer.word_index)+1, self.encoder_embed_dim, self.encoder_units, self.batch_size)
        decoder = Decoder(len(self.trg_tokenizer.word_index)+1, self.decoder_embed_dim, self.decoder_units, self.batch_size)
        
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        
        return encoder, decoder, optimizer, checkpoint
    
    def predict(self, src_tensor, token_separator=''):
        encoder, decoder, optimizer, checkpoint = self.init_network()
        
        checkpoint.restore(tf.train.latest_checkpoint('outputs/nmt_models'))
        n = len(src_tensor)
        
        results = np.zeros((n, self.max_length_trg))

        hidden = [tf.zeros((n, self.encoder_units))]
        enc_out, enc_hidden = encoder(src_tensor, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.trg_tokenizer.word_index[self.start_token]]*n, 1)

        for t in range(self.max_length_trg):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out, is_training=False)
            predicted_ids = tf.argmax(predictions, axis=1).numpy()
            
            results[:,t] = predicted_ids

            dec_input = tf.expand_dims(predicted_ids, 1)
        
        outputs = []
        for res in results:
            result = ''
            for x in res:
                token = self.trg_tokenizer.index_word[x]
                if token == self.end_token:
                    break
                result += str(token) + str(token_separator)
            
            outputs.append(result)

        return outputs
        
    def fit(self):
        encoder, decoder, optimizer, checkpoint = self.init_network()
        best_loss = float("Inf")
        
        for epoch in range(self.num_epochs):
            train_iter = self.data_generator(self.src_tensor, self.trg_tensor, self.batch_size)
            steps_per_epoch=int(math.ceil(float(len(self.src_tensor))/self.batch_size))

            total_loss = 0
            
            for batch in range(steps_per_epoch):
                src, trg = next(train_iter)
                batch_size = src.shape[0]
                batch_loss = train_step(src, trg, self.trg_tokenizer, encoder, decoder, optimizer, 
                                        batch_size, self.start_token)
                total_loss += batch_loss
                    
            print('Epoch {} Mean Training Loss {:.8f}'.format(epoch + 1, total_loss/len(self.src_tensor)))
            
            if total_loss < best_loss:
                best_loss = total_loss
                checkpoint.save(file_prefix = 'outputs/nmt_models/nmt_model.h5')