import numpy as np
import math
from collections import defaultdict
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import f1_score
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
    fn_each = lambda x: max(0.01 * x, 0.99 * x)
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def output_layer_activation_softmax(inputs):
    fn_each = lambda x: math.exp(x)
    vectorize = np.vectorize(fn_each)

    out = vectorize(inputs)

    return out/np.sum(out)

def output_layer_grad_softmax(pred_outs, true_outs):
    fn_each = lambda x, y: x - y
    vectorize = np.vectorize(fn_each)

    return vectorize(pred_outs, true_outs)

def hidden_layer_grad_sigmoid(inputs):
    fn_each = lambda x: x * (1-x)
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def hidden_layer_grad_relu(inputs):
    fn_each = lambda x: 0.99 if x > 0 else 0.01
    vectorize = np.vectorize(fn_each)

    return vectorize(inputs)

def loss_cross_entropy(preds, actuals):
    fn_each = lambda x, y: -y * math.log(x, 2)
    vectorize = np.vectorize(fn_each)

    return np.sum(vectorize(preds, actuals))

def loss(outputs, targets, num_layers):
    total_loss = 0.0

    for row in outputs:
        predictions = outputs[row][num_layers-1]
        total_loss += loss_cross_entropy(predictions, targets[row,])

    return total_loss

def forward_pass(trainX, layers, weights, biases):

    nested_dict = lambda: defaultdict(nested_dict)
    outputs = nested_dict()

    for row in range(trainX.shape[0]):
        input = trainX[row,]

        for layer in range(len(layers)):
            weight_matrix = weights[layer].T
            node_inp = weight_matrix.dot(input) + biases[layer]

            if layer == len(layers)-1:
                outputs[row][layer] = output_layer_activation_softmax(node_inp)
            else:
                outputs[row][layer] = hidden_layer_activation_relu(node_inp)

            input = outputs[row][layer]

    return outputs

def error_backpropagation(trainX, trainY, outputs, layers, weights, biases, momentums, learning_rate, momentum_rate):

    nested_dict = lambda: defaultdict(nested_dict)
    bp_grads = nested_dict()

    for layer in reversed(range(len(layers))):

        layer_weights = weights[layer]
        layer_biases = biases[layer]
        layer_momentums = momentums[layer]

        for row in outputs:
            if layer == len(layers) - 1:
                bp_grads[row][layer] = output_layer_grad_softmax(outputs[row][layer], trainY[row,])
            else:
                bp_grads[row][layer] = hidden_layer_grad_relu(outputs[row][layer])

                next_layer_weights = weights[layer + 1]

                for i in range(next_layer_weights.shape[0]):
                    total_grad = 0.0
                    for j in range(next_layer_weights.shape[1]):
                        total_grad += bp_grads[row][layer+1][j] * next_layer_weights[i, j]

                    bp_grads[row][layer][i] *= total_grad

        for i in range(layer_weights.shape[0]):
            for j in range(layer_weights.shape[1]):
                total_err = 0.0
                for row in bp_grads:
                    if layer > 0:
                        total_err += bp_grads[row][layer][j] * outputs[row][layer - 1][i]
                    else:
                        total_err += bp_grads[row][layer][j] * trainX[row, i]

                layer_momentums[i, j] = momentum_rate * layer_momentums[i, j] + learning_rate * total_err
                layer_weights[i, j] -= layer_momentums[i, j]

        for i in range(layer_weights.shape[1]):
            total_err = 0.0
            for row in bp_grads:
                total_err += bp_grads[row][layer][i]

            layer_biases[i] -= learning_rate * total_err

        weights[layer] = layer_weights
        biases[layer] = layer_biases
        momentums[layer] = layer_momentums

    return weights, biases, momentums


def train_neural_network(trainX, trainY, hidden_layers=[5, 2],
                         num_epochs=10, learning_rate=0.0005, train_batch_size=32, momentum_rate=0.9):

    num_classes = len(set(trainY))

    trainY = one_hot_encoding(trainY)
    layers = hidden_layers + [num_classes]

    weights, biases, momentums = dict(), dict(), dict()

    for layer in range(len(layers)):
        if layer == 0:
            num_rows = trainX.shape[1]
            num_cols = layers[layer]
        else:
            num_rows = layers[layer-1]
            num_cols = layers[layer]

        weights[layer] = np.random.normal(0.0, 0.5, num_rows*num_cols).reshape(num_rows, num_cols)
        momentums[layer] = np.zeros((num_rows, num_cols))
        biases[layer] = np.random.normal(0.0, 0.5, num_cols)

    trainX_batches, trainY_batches = generate_batches(trainX, trainY, train_batch_size)

    prev_loss = 0

    for epoch in range(num_epochs):

        for batch in range(len(trainX_batches)):
            trainX_batch = trainX_batches[batch]
            trainY_batch = trainY_batches[batch]

            outputs = forward_pass(trainX_batch, layers, weights, biases)

            weights, biases, momentums = error_backpropagation(trainX_batch, trainY_batch, outputs, layers, weights,
                                                               biases,
                                                               momentums, learning_rate, momentum_rate)

        outputs = forward_pass(trainX, layers, weights, biases)
        curr_loss = loss(outputs, trainY, len(layers))

        if epoch > 0 and math.fabs(curr_loss-prev_loss)/float(prev_loss) < 0.01:
            break

        prev_loss = curr_loss

    model = (weights, biases, layers)

    return model

def predict_neural_network(testX, model, type="class"):
    weights, biases, layers = model
    outputs = forward_pass(testX, layers, weights, biases)

    outs = []

    for row in outputs:
        preds = outputs[row][len(layers)-1]
        if type == "class":
            outs += [np.argmax(preds)]
        else:
            outs += [preds]

    return outs

def train_nn_cv(trainX, trainY, hidden_layers=[5, 2],
                num_epochs=10, learning_rate=0.0005, train_batch_size=32, momentum_rate=0.9, num_cv=5):

    kf = KFold(n_splits=num_cv)

    for train_index, test_index in kf.split(trainX):
        trainX_batch, testX_batch = trainX[train_index], trainX[test_index]
        trainY_batch, testY_batch = trainY[train_index], trainY[test_index]

        model = train_neural_network(trainX_batch, trainY_batch, hidden_layers=hidden_layers,
                                     learning_rate=learning_rate, num_epochs=num_epochs,
                                     train_batch_size=train_batch_size, momentum_rate=momentum_rate)

        preds = predict_neural_network(testX_batch, model)
        print f1_score(testY_batch, preds, average='weighted')



mydata = datasets.load_breast_cancer()

trainX = mydata.data
trainX = normalize(trainX, axis=0, norm='max')

trainY = mydata.target

train_nn_cv(trainX, trainY, hidden_layers=[15], learning_rate=0.0005, num_epochs=100,
            train_batch_size=32, momentum_rate=0.9, num_cv=3)