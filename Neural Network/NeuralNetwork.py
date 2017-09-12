import numpy as np
import math
from collections import defaultdict
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold

def one_hot_encoding(classes):
    num_classes = len(set(classes))
    targets = np.array([classes]).reshape(-1)

    return np.eye(num_classes)[targets]

def generate_batches(trainX, trainY, batch_size):
    concatenated = np.column_stack((trainX, trainY))
    np.random.shuffle(concatenated)

    trainX = concatenated[:,:trainX.shape[1]]
    trainY = concatenated[:,trainX.shape[1]:]

    num_batches = math.ceil(float(trainX.shape[0])/batch_size)

    return np.array_split(trainX, num_batches), np.array_split(trainY, num_batches)

def hidden_layer_activation_sigmoid(inputs):
    fn_each = lambda x: float(1.0)/(1.0 + math.exp(-x))
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def hidden_layer_activation_relu(inputs):
    fn_each = lambda x: max(0.00001 * x, 0.99999 * x)
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def output_layer_activation_softmax(inputs):
    fn_each = lambda x: math.exp(x) if x <= 5 else math.exp(5)
    vectorize = np.vectorize(fn_each)

    inputs = (inputs.T - np.mean(inputs, axis=1)).T

    out = vectorize(inputs)

    return (out.T/np.sum(out, axis=1)).T

def output_layer_grad_softmax(pred_outs, true_outs):
    fn_each = lambda x, y: x - y
    vectorize = np.vectorize(fn_each)

    return vectorize(pred_outs, true_outs)

def hidden_layer_grad_sigmoid(inputs):
    fn_each = lambda x: x * (1-x)
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def hidden_layer_grad_relu(inputs):
    fn_each = lambda x: 0.99999 if x > 0 else 0.00001
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def loss_cross_entropy(preds, actuals):
    fn_each = lambda x, y: -y * math.log(x, 2)
    vectorize = np.vectorize(fn_each)

    return np.sum(vectorize(preds, actuals))

def loss(outputs, targets, num_layers):

    predictions = outputs[num_layers - 1]
    total_loss = loss_cross_entropy(predictions, targets)

    return total_loss

def forward_pass(trainX, layers, weights, biases, dropout_rate):

    nested_dict = lambda: defaultdict(nested_dict)
    outputs = nested_dict()

    curr_input = trainX

    for layer in range(len(layers)):
        linear_inp = curr_input.dot(weights[layer]) + biases[layer]

        if layer == len(layers) - 1:
            outputs[layer] = output_layer_activation_softmax(linear_inp)
        else:
            binomial_mat = np.zeros(shape=(trainX.shape[0], weights[layer].shape[1]))
            for row in range(trainX.shape[0]):
                binomial_mat[row,] = np.random.binomial(1, 1 - dropout_rate, weights[layer].shape[1])

            outputs[layer] = hidden_layer_activation_relu(linear_inp) * binomial_mat

        curr_input = outputs[layer]

    return outputs

def error_backpropagation(trainX, trainY, outputs, layers, weights, biases, momentums, learning_rate, momentum_rate):

    nested_dict = lambda: defaultdict(nested_dict)
    bp_grads = nested_dict()

    for layer in reversed(range(len(layers))):

        if layer == len(layers) - 1:
            bp_grads[layer] = output_layer_grad_softmax(outputs[layer], trainY)
        else:
            bp_grads[layer] = hidden_layer_grad_relu(outputs[layer])

            next_layer_weights = weights[layer + 1]

            for i in range(next_layer_weights.shape[0]):
                total_grad = np.sum(bp_grads[layer + 1] * next_layer_weights[i, ], axis=1)

                bp_grads[layer][:, i] *= total_grad

        if layer > 0:
            total_err = outputs[layer - 1].T.dot(bp_grads[layer])
        else:
            total_err = trainX.T.dot(bp_grads[layer])

        momentums[layer] = momentum_rate * momentums[layer] + learning_rate * total_err * float(1.0) / trainX.shape[0]
        weights[layer] -= momentums[layer]

        biases[layer] -= learning_rate * np.sum(bp_grads[layer], axis=0) * float(1.0) / trainX.shape[0]

    return weights, biases, momentums

def initialize(layers, num_features):
    weights, biases, momentums = dict(), dict(), dict()

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

    return weights, biases, momentums

def scale_weights_dropout(weights, biases, dropout_rate):

    scaled_weights, scaled_biases = dict(), dict()

    for layer in weights:
        scaled_weights[layer] = weights[layer] * (1 - dropout_rate)
        scaled_biases[layer] = biases[layer] * (1 - dropout_rate)

    return scaled_weights, scaled_biases

def constrain_weights(weights, biases, constrain_radius):

    constrained_weights, constrained_biases = dict(), dict()

    for layer in weights:
        v1 = float(constrain_radius) / np.sqrt(np.sum(np.square(weights[layer]), axis=0))
        v2 = float(constrain_radius) / np.sqrt(np.sum(np.square(biases[layer])))

        constrained_weights[layer] = weights[layer] * v1
        constrained_biases[layer] = biases[layer] * v2

    return constrained_weights, constrained_biases

def train_neural_network(trainX, trainY, hidden_layers=[5, 2],
                         num_epochs=10, learning_rate=0.0005, train_batch_size=32, momentum_rate=0.9,
                         dropout_rate=0.2, constrain_radius=3.0):

    num_classes = len(set(trainY))

    trainY = one_hot_encoding(trainY)
    layers = hidden_layers + [num_classes]

    weights, biases, momentums = initialize(layers, trainX.shape[1])

    weights, biases = constrain_weights(weights, biases, constrain_radius)

    trainX_batches, trainY_batches = generate_batches(trainX, trainY, train_batch_size)

    losses = []

    for epoch in range(num_epochs):

        for batch in range(len(trainX_batches)):
            trainX_batch = trainX_batches[batch]
            trainY_batch = trainY_batches[batch]

            outputs = forward_pass(trainX_batch, layers, weights, biases, dropout_rate)

            weights, biases, momentums = error_backpropagation(trainX_batch, trainY_batch, outputs, layers, weights,
                                                               biases, momentums, learning_rate, momentum_rate)

            weights, biases = constrain_weights(weights, biases, constrain_radius)

        dummy_weights, dummy_biases = scale_weights_dropout(weights, biases, dropout_rate)

        outputs = forward_pass(trainX, layers, dummy_weights, dummy_biases, dropout_rate)

        curr_loss = loss(outputs, trainY, len(layers))

        cond_1 = curr_loss <= trainX.shape[0] * (-math.log(0.9, 2))
        cond_2 = len(losses) > 2 and curr_loss > losses[-1] > losses[-2] > losses[-3]

        if epoch > 0 and (cond_1 or cond_2):
            break

        losses.append(curr_loss)

    weights, biases = scale_weights_dropout(weights, biases, dropout_rate)

    model = (weights, biases, layers, dropout_rate)

    return model

def predict_neural_network(testX, model, type="class"):

    weights, biases, layers, dropout_rate = model
    outputs = forward_pass(testX, layers, weights, biases, dropout_rate)

    preds = outputs[len(layers) - 1]
    outs = []

    for row in range(preds.shape[0]):
        if type == "class":
            outs += [np.argmax(preds[row,])]
        else:
            outs += [preds[row,]]

    return outs

def train_nn_cv(trainX, trainY, hidden_layers=[5, 2],
                num_epochs=10, learning_rate=0.0005, train_batch_size=32, momentum_rate=0.9,
                dropout_rate=0.2, constrain_radius=3.0, num_cv=5):

    kf = KFold(n_splits=num_cv)

    for train_index, test_index in kf.split(trainX):
        trainX_batch, testX_batch = trainX[train_index], trainX[test_index]
        trainY_batch, testY_batch = trainY[train_index], trainY[test_index]

        train_mean = np.mean(trainX_batch, axis=0)
        train_sd = np.std(trainX_batch, axis=0)

        trainX_batch = (trainX_batch-train_mean)/train_sd
        testX_batch = (testX_batch-train_mean)/train_sd

        model = train_neural_network(trainX_batch, trainY_batch, hidden_layers=hidden_layers,
                                     learning_rate=learning_rate, num_epochs=num_epochs,
                                     train_batch_size=train_batch_size, momentum_rate=momentum_rate,
                                     dropout_rate=dropout_rate, constrain_radius=constrain_radius)

        preds_train = predict_neural_network(trainX_batch, model)
        preds_test = predict_neural_network(testX_batch, model)

        print "Train F1-Score = ", f1_score(trainY_batch, preds_train, average='weighted')
        print "Train Accuracy = ", accuracy_score(trainY_batch, preds_train)

        print "Validation F1-Score = ", f1_score(testY_batch, preds_test, average='weighted')
        print "Validation Accuracy = ", accuracy_score(testY_batch, preds_test)


mydata = datasets.load_breast_cancer()

trainX = mydata.data
trainY = mydata.target

train_nn_cv(trainX, trainY, hidden_layers=[15, 10], learning_rate=0.05, num_epochs=100,
            train_batch_size=50, momentum_rate=0.9, dropout_rate=0.1, constrain_radius=3.0, num_cv=5)