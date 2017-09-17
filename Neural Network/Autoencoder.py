import numpy as np
import math
from collections import defaultdict
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold


def standardize_mean_var(mydata, mean=None, var=None):

    if mean is None and var is None:
        mean = np.mean(mydata, axis=0)
        var = np.var(mydata, axis=0)

    std_data = (mydata - mean) * (var + 0.5) ** -0.5

    return std_data, mean, var


def hidden_layer_activation_sigmoid(inputs):
    fn_each = lambda x: float(1.0)/(1.0 + math.exp(-x))
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)


def hidden_layer_activation_relu(inputs):
    fn_each = lambda x: max(0.00001 * x, 0.99999 * x)
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)


def hidden_layer_grad_sigmoid(inputs):
    fn_each = lambda x: x * (1-x)
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)


def hidden_layer_grad_relu(inputs):
    fn_each = lambda x: 0.99999 if x > 0 else 0.00001
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)


def loss_mse(preds, actuals):
    fn_each = lambda x, y: 0.5 * (x - y)**2
    vectorize = np.vectorize(fn_each)

    return np.sum(vectorize(preds, actuals))


def loss(outputs, targets):
    num_layers = len(outputs)

    predictions = outputs[num_layers - 1]
    total_loss = loss_mse(predictions, targets)

    return total_loss


def initialize(layers, num_features):
    weights, biases, momentums, gamma, beta = dict(), dict(), dict(), dict(), dict()

    for layer in range(len(layers)):
        if layer == 0:
            num_rows = num_features
            num_cols = layers[layer]
        else:
            num_rows = layers[layer - 1]
            num_cols = layers[layer]

        weights[layer] = np.random.normal(0.0, 1.0, num_rows * num_cols).reshape(num_rows, num_cols) / math.sqrt(
            2.0 * num_rows)
        momentums[layer] = np.zeros((num_rows, num_cols))
        biases[layer] = np.random.normal(0.0, 1.0, num_cols)

        gamma[layer] = np.ones(num_cols)
        beta[layer] = np.zeros(num_cols)

    return weights, biases, momentums, gamma, beta


def constrain_weights(layer_weights, layer_biases, constrain_radius):

    v1 = float(constrain_radius) / np.sqrt(np.sum(np.square(layer_weights), axis=0))
    v2 = float(constrain_radius) / np.sqrt(np.sum(np.square(layer_biases)))

    return layer_weights * v1, layer_biases * v2


def train_forward_pass(curr_input, layer_weights, layer_biases, layer_gamma, layer_beta,
                       d_layer_weights, d_layer_biases, d_layer_gamma, d_layer_beta):

    linear_inp_h = curr_input.dot(layer_weights) + layer_biases

    scaled_linear_inp_h, mean_linear_inp_h, var_linear_inp_h = standardize_mean_var(linear_inp_h)

    shifted_inp_h = layer_gamma * scaled_linear_inp_h + layer_beta

    output_h = hidden_layer_activation_relu(shifted_inp_h)

    linear_inp_o = output_h.dot(d_layer_weights) + d_layer_biases

    scaled_linear_inp_o, mean_linear_inp_o, var_linear_inp_o = standardize_mean_var(linear_inp_o)

    shifted_inp_o = d_layer_gamma * scaled_linear_inp_o + d_layer_beta

    output_o = output_layer_activation_reg(shifted_inp_o)

    hidden_layer_data = (linear_inp_h, scaled_linear_inp_h, mean_linear_inp_h, var_linear_inp_h, output_h)
    output_layer_data = (linear_inp_o, scaled_linear_inp_o, mean_linear_inp_o, var_linear_inp_h, output_o)

    return hidden_layer_data, output_layer_data


def train_autoencoder(trainX,
                      hidden_layers=[5, 2],
                      num_epochs=10,
                      weights_learning_rate=0.5,
                      bn_learning_rate=0.5,
                      momentum_rate=0.9,
                      constrain_radius=3.0,
                      sparsity=0.05):

    layers = hidden_layers

    weights, biases, momentums, gamma, beta = initialize(layers, trainX.shape[1])

    expected_mean_linear_inp, expected_var_linear_inp = dict(), dict()

    for layer in range(len(layers)):
        expected_mean_linear_inp[layer] = np.zeros(weights[layer].shape[1])
        expected_var_linear_inp[layer] = np.zeros(weights[layer].shape[1])

    curr_input = trainX
    hidden_layer_output = curr_input

    for layer in range(len(layers)):

        sub_layers = [layers[layer]] + [len(curr_input)]

        s_weights, s_biases, s_momentums, s_gamma, s_beta = initialize(sub_layers, curr_input.shape[1])

        s_weights, s_biases = constrain_weights(s_weights, s_biases, constrain_radius)

        losses = []

        for epoch in range(num_epochs):

            fwd_pass_data = train_forward_pass(curr_input, s_weights, s_biases, s_gamma, s_beta)

            outputs, linear_inp, scaled_linear_inp, mean_linear_inp, var_linear_inp = fwd_pass_data

            expected_mean_linear_inp[layer] += mean_linear_inp[layer]

            expected_var_linear_inp[layer] += var_linear_inp[layer]

            backprop = error_backpropagation(curr_input,
                                             output=output,
                                             linear_inp=linear_inp[layer],
                                             scaled_linear_inp=scaled_linear_inp[layer],
                                             mean_linear_inp=mean_linear_inp[layer],
                                             var_linear_inp=var_linear_inp[layer],
                                             weights=weights[layer],
                                             biases=biases[layer],
                                             momentums=momentums[layer],
                                             gamma=gamma[layer],
                                             beta=beta[layer],
                                             d_weights=d_weights,
                                             d_biases=d_biases,
                                             d_momentums=d_momentums,
                                             d_gamma=d_gamma,
                                             d_beta=d_beta,
                                             bn_learning_rate=bn_learning_rate,
                                             weights_learning_rate=weights_learning_rate,
                                             momentum_rate=momentum_rate)

            weights[layer], biases[layer], momentums[layer], gamma[layer], beta[
                layer], d_weights, d_biases, d_momentums, d_gamma, d_beta = backprop

            weights[layer], biases[layer] = constrain_weights(weights[layer], biases[layer], constrain_radius)

            d_weights, d_biases = constrain_weights(d_weights, d_biases, constrain_radius)

            output = test_forward_pass(curr_input,
                                       weights=weights[layer],
                                       biases=biases[layer],
                                       gamma=gamma[layer],
                                       beta=beta[layer],
                                       d_weights=d_weights,
                                       d_biases=d_biases,
                                       d_gamma=d_gamma,
                                       d_beta=d_beta,
                                       mean_linear_inp=exp_mean_linear_inp[layer],
                                       var_linear_inp=exp_var_linear_inp[layer])

            curr_loss = loss_reg(output, curr_input)

            cond_1 = curr_loss <= trainX.shape[0] * 0.001
            cond_2 = len(losses) > 2 and curr_loss > losses[-1] > losses[-2] > losses[-3]

            if epoch > 0 and (cond_1 or cond_2):
                break

            losses.append(curr_loss)

        curr_input = hidden_layer_output

        m = trainX.shape[0]

        expected_mean_linear_inp[layer] /= m
        expected_var_linear_inp[layer] = (float(m) / (m - 1)) * expected_var_linear_inp[layer] / m

    model = (weights, biases, gamma, beta, exp_mean_linear_inp, exp_var_linear_inp)

    return model

