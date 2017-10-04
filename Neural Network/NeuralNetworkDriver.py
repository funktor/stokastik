import NeuralNetwork
import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score, accuracy_score

def digits_classification():
    mydata = datasets.load_digits()

    trainX = mydata.data
    trainY = mydata.target

    n_samples = trainX.shape[0]

    X_train, y_train = trainX[:int(.8 * n_samples)], trainY[:int(.8 * n_samples)]
    X_test, y_test = trainX[int(.8 * n_samples):], trainY[int(.8 * n_samples):]

    hidden_layers = [1000]
    weights_learning_rate = 0.5
    bn_learning_rate = 0.9
    num_epochs = 100
    train_batch_size = 32
    momentum_rate = 0.95
    dropout_rate = 0.0

    layers = hidden_layers + [len(set(y_train))]
    weights, biases, momentums, gamma, beta = NeuralNetwork.initialize(layers, X_train.shape[1])

    NeuralNetwork.train_nn_cv(X_train, y_train,
                              hidden_layers=hidden_layers,
                              weights_learning_rate=weights_learning_rate,
                              bn_learning_rate=bn_learning_rate,
                              num_epochs=num_epochs,
                              train_batch_size=train_batch_size,
                              momentum_rate=momentum_rate,
                              dropout_rate=dropout_rate,
                              num_cv=3,
                              ini_weights=weights,
                              ini_biases=biases,
                              ini_momentums=momentums,
                              ini_gamma=gamma,
                              ini_beta=beta)

    nn_model = NeuralNetwork.train_neural_network(X_train, y_train,
                                                  hidden_layers=hidden_layers,
                                                  weights_learning_rate=weights_learning_rate,
                                                  bn_learning_rate=bn_learning_rate,
                                                  num_epochs=num_epochs,
                                                  train_batch_size=train_batch_size,
                                                  momentum_rate=momentum_rate,
                                                  dropout_rate=dropout_rate,
                                                  ini_weights=weights,
                                                  ini_biases=biases,
                                                  ini_momentums=momentums,
                                                  ini_gamma=gamma,
                                                  ini_beta=beta,
                                                  type="classification")

    nn_predict = NeuralNetwork.predict_neural_network(X_test, nn_model, type="classification")

    print "Test F1-Score = ", f1_score(y_test, nn_predict, average='weighted')
    print "Test Accuracy = ", accuracy_score(y_test, nn_predict)

    print ""

    autoencoder_model = NeuralNetwork.train_autoencoder_reg(X_train, y_train,
                                                            hidden_layers=hidden_layers,
                                                            weights_learning_rate=weights_learning_rate,
                                                            bn_learning_rate=bn_learning_rate,
                                                            num_epochs=num_epochs,
                                                            train_batch_size=train_batch_size,
                                                            momentum_rate=momentum_rate,
                                                            dropout_rate=dropout_rate,
                                                            ini_weights=weights,
                                                            ini_biases=biases,
                                                            ini_momentums=momentums,
                                                            ini_gamma=gamma,
                                                            ini_beta=beta,
                                                            type="classification")

    nn_predict = NeuralNetwork.predict_neural_network(X_test, autoencoder_model, type="classification")

    print "Test F1-Score = ", f1_score(y_test, nn_predict, average='weighted')
    print "Test Accuracy = ", accuracy_score(y_test, nn_predict)

    print ""


def xor_gate():

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    hidden_layers = [10, 10]
    weights_learning_rate = 0.5
    bn_learning_rate = 0.9
    num_epochs = 100
    train_batch_size = 32
    momentum_rate = 0.95
    dropout_rate = 0.0

    layers = hidden_layers + [1]
    weights, biases, momentums, gamma, beta = NeuralNetwork.initialize(layers, X_train.shape[1])

    nn_model = NeuralNetwork.train_neural_network(X_train, y_train,
                                                  hidden_layers=hidden_layers,
                                                  weights_learning_rate=weights_learning_rate,
                                                  bn_learning_rate=bn_learning_rate,
                                                  num_epochs=num_epochs,
                                                  train_batch_size=train_batch_size,
                                                  momentum_rate=momentum_rate,
                                                  dropout_rate=dropout_rate,
                                                  ini_weights=weights,
                                                  ini_biases=biases,
                                                  ini_momentums=momentums,
                                                  ini_gamma=gamma,
                                                  ini_beta=beta,
                                                  type="regression")

    nn_predict = NeuralNetwork.predict_neural_network(X_train, nn_model, type="regression")
    print nn_predict
    print ""

    autoencoder_model = NeuralNetwork.train_autoencoder_reg(X_train, y_train,
                                                            hidden_layers=hidden_layers,
                                                            weights_learning_rate=weights_learning_rate,
                                                            bn_learning_rate=bn_learning_rate,
                                                            num_epochs=num_epochs,
                                                            train_batch_size=train_batch_size,
                                                            momentum_rate=momentum_rate,
                                                            dropout_rate=dropout_rate,
                                                            ini_weights=weights,
                                                            ini_biases=biases,
                                                            ini_momentums=momentums,
                                                            ini_gamma=gamma,
                                                            ini_beta=beta,
                                                            type="regression")

    nn_predict = NeuralNetwork.predict_neural_network(X_train, autoencoder_model, type="regression")
    print nn_predict
    print ""


def custom_fun():

    func = lambda x, y: ((3*x+2*y)**3 + (x+y)**2)**2 + ((3*x+2*y)**2 + (x-y)**3)**2

    a = np.random.uniform(0.0, 0.5, 100)
    b = np.random.uniform(0.0, 0.5, 100)
    c = [func(x, y) for (x, y) in zip(a, b)]

    trainX = np.array([a, b]).T
    trainY = np.array(c)

    n_samples = trainX.shape[0]

    X_train, y_train = trainX[:int(.8 * n_samples)], trainY[:int(.8 * n_samples)]
    X_test, y_test = trainX[int(.8 * n_samples):], trainY[int(.8 * n_samples):]

    print y_test

    hidden_layers = [500, 300, 100]
    weights_learning_rate = 0.001
    bn_learning_rate = 0.9
    num_epochs = 1000
    train_batch_size = 100
    momentum_rate = 0.95
    dropout_rate = 0.0

    layers = hidden_layers + [1]
    weights, biases, momentums, gamma, beta = NeuralNetwork.initialize(layers, X_train.shape[1])

    nn_model = NeuralNetwork.train_neural_network(X_train, y_train,
                                                  hidden_layers=hidden_layers,
                                                  weights_learning_rate=weights_learning_rate,
                                                  bn_learning_rate=bn_learning_rate,
                                                  num_epochs=num_epochs,
                                                  train_batch_size=train_batch_size,
                                                  momentum_rate=momentum_rate,
                                                  dropout_rate=dropout_rate,
                                                  ini_weights=weights,
                                                  ini_biases=biases,
                                                  ini_momentums=momentums,
                                                  ini_gamma=gamma,
                                                  ini_beta=beta,
                                                  type="regression")

    nn_predict = NeuralNetwork.predict_neural_network(X_test, nn_model, type="regression")
    print nn_predict
    print NeuralNetwork.loss_reg(nn_predict, y_test)
    print ""

    autoencoder_model = NeuralNetwork.train_autoencoder_reg(X_train, y_train,
                                                            hidden_layers=hidden_layers,
                                                            weights_learning_rate=weights_learning_rate,
                                                            bn_learning_rate=bn_learning_rate,
                                                            num_epochs=num_epochs,
                                                            train_batch_size=train_batch_size,
                                                            momentum_rate=momentum_rate,
                                                            dropout_rate=dropout_rate,
                                                            ini_weights=weights,
                                                            ini_biases=biases,
                                                            ini_momentums=momentums,
                                                            ini_gamma=gamma,
                                                            ini_beta=beta,
                                                            type="regression")

    nn_predict = NeuralNetwork.predict_neural_network(X_test, autoencoder_model, type="regression")
    print nn_predict
    print NeuralNetwork.loss_reg(nn_predict, y_test)
    print ""

custom_fun()