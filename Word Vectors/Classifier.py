import Doc2Vec, Utilities
import numpy as np
from sklearn import svm

def trainTest(trainData, trainLabels, testData, testLabels):
    clf = svm.SVC(decision_function_shape='ovo', C=100, gamma=0.9, kernel='rbf')
    clf.fit(trainData, trainLabels)

    return clf.score(testData, testLabels)

def constructDocArrayFromWords(tokens, vocab, vectorizer, vectorModel, docFeatureMat):
    docArrays = np.zeros((len(tokens), Doc2Vec.vector_size))

    for i in range(len(tokens)):
        fileTokens = tokens[i]
        temp = np.zeros((len(fileTokens), Doc2Vec.vector_size))
        weights = np.zeros(len(fileTokens))

        for j in range(len(fileTokens)):
            token = fileTokens[j]

            if token in vocab:
                word_vector = vectorModel[token]
                feature_index = vectorizer.vocabulary_.get(token)
                tfidf = docFeatureMat[i, feature_index]
            else:
                word_vector = np.zeros(Doc2Vec.vector_size)
                tfidf = 0

            temp[j] = np.array(word_vector)
            weights[j] = tfidf

        weightSum = np.sum(weights)

        if (weightSum > 0):
            weights = np.array([weight / weightSum for weight in weights])

        docArrays[i] = weights.dot(temp)

    return docArrays


def trainTestSVM(train, test):

    vectorizer = Utilities.getVectorizer()

    X_train = vectorizer.fit_transform(train['Contents'])
    X_test = vectorizer.transform(test['Contents'])

    return trainTest(X_train, train['Labels'], X_test, test['Labels'])

def trainTestSVM_Doc2Vec(train, test, useFullData=1):

    if (useFullData == 1):
        tokens = train['Tokens'] + test['Tokens']
    else:
        tokens = train['Tokens']

    vectorModel = Doc2Vec.trainDoc2Vec(tokens, 'doc2vec')

    trainTokens = train['Tokens']

    trainLabels = train['Labels']

    trainArrays = np.zeros((len(trainTokens), Doc2Vec.vector_size))

    for i in range(len(trainTokens)):
        trainArrays[i] = vectorModel.docvecs['DOC_' + str(i)]

    testTokens = test['Tokens']

    testLabels = test['Labels']

    testArrays = np.zeros((len(testTokens), Doc2Vec.vector_size))

    for i in range(len(testTokens)):
        if (useFullData == 1):
            testArrays[i] = vectorModel.docvecs['DOC_' + str(i + len(trainTokens))]
        else:
            testArrays[i] = vectorModel.infer_vector(testTokens[i], steps=10)

    return trainTest(trainArrays, trainLabels, testArrays, testLabels)


def trainTestSVM_Word2Vec(train, test):

    vectorizer = Utilities.getVectorizer()

    X_train = vectorizer.fit_transform(train['Contents'])
    X_test = vectorizer.transform(test['Contents'])

    tokens = train['Tokens'] + test['Tokens']

    vectorModel = Doc2Vec.trainDoc2Vec(tokens, 'doc2vec')

    vocab = set.intersection(set(vectorModel.wv.vocab), set(vectorizer.vocabulary_.keys()))

    trainTokens = train['Tokens']

    trainLabels = train['Labels']

    trainArrays = constructDocArrayFromWords(trainTokens, vocab, vectorizer, vectorModel, X_train)

    testTokens = test['Tokens']

    testLabels = test['Labels']

    testArrays = constructDocArrayFromWords(testTokens, vocab, vectorizer, vectorModel, X_test)

    return trainTest(trainArrays, trainLabels, testArrays, testLabels)


train = Doc2Vec.getTrainTokens()
test = Doc2Vec.getTestTokens()

print trainTestSVM_Word2Vec(train, test)
