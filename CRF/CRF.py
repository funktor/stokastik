import nltk, os, eli5, collections, pickle, re
from sklearn_crfsuite import metrics
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from nltk.corpus import brown
import numpy as np


def wordShapeFeatures(word, wordId):
    return {
        wordId + '_is_title': word.istitle(),
        wordId + '_is_lower': word.islower(),
        wordId + '_is_upper': word.isupper(),
        wordId + '_is_digit': word.isdigit(),
        wordId + '_is_camelcase': re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word) is not None,
        wordId + '_is_abbv': re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word) is not None,
        wordId + '_has_hyphen': re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word) is not None,
    }


def wordProperties(word, wordId):

    stemmer = SnowballStemmer('english')

    return {
        wordId + '_stemmed': stemmer.stem(word),
        wordId + '_prefix-1': word[0],
        wordId + '_prefix-2': word[:2],
        wordId + '_prefix-3': word[:3],
        wordId + '_suffix-1': word[-1],
        wordId + '_suffix-2': word[-2:],
        wordId + '_suffix-3': word[-3:],
        wordId + '_lower': word.lower(),
    }


def features(sentence, index):

    currWord = sentence[index][0]

    if index > 0:
        prevWord = sentence[index - 1][0]
    else:
        prevWord = '<START>'

    if index < len(sentence)-1:
        nextWord = sentence[index + 1][0]
    else:
        nextWord = '<END>'

    currWordShape = wordShapeFeatures(currWord, 'currWd')
    prevWordShape = wordShapeFeatures(prevWord, 'prevWd')
    nextWordShape = wordShapeFeatures(nextWord, 'nextWd')

    currWordProp = wordProperties(currWord, 'currWd')
    prevWordProp = wordProperties(prevWord, 'prevWd')
    nextWordProp = wordProperties(nextWord, 'nextWd')

    outFeatures = {
        'word': currWord,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prev_word': prevWord,
        'next_word': nextWord,
    }

    outFeatures.update(currWordShape)
    outFeatures.update(prevWordShape)
    outFeatures.update(nextWordShape)

    outFeatures.update(currWordProp)
    outFeatures.update(prevWordProp)
    outFeatures.update(nextWordProp)

    return outFeatures


def getFeatureMatrix(sentences):
    feats = []

    for sent in sentences:
        for index in range(len(sent)):
            feats.append(features(sent, index))

    vectorizer = DictVectorizer(sparse=False)
    featureMatrix = vectorizer.fit_transform(feats)

    featureMatrix[featureMatrix > 0] = 1

    return np.array(featureMatrix)


def transformDatasetSequence(sentences):
    wordFeatures, wordLabels = [], []

    for sent in sentences:
        feats, labels = [], []

        for index in range(len(sent)):
            feats.append(features(sent, index))
            labels.append(sent[index][1])

        wordFeatures.append(feats)
        wordLabels.append(labels)

    return wordFeatures, wordLabels


def forward_pass(sentence, num_labels, featureMatrix):
    forward_score = np
def train_crf(trainSentences, featureMatrix):
    weights = np.random.normal(0.0, 0.5, featureMatrix.shape[1])

    row_sums = np.sum(featureMatrix, axis=1)
    col_sums = np.sum(featureMatrix, axis=0)

    expectation_data = col_sums / len(trainSentences)




brown_tagged_sents = brown.tagged_sents(categories='news')

size = int(len(brown_tagged_sents) * 0.7)

featureMatrix = getFeatures(brown_tagged_sents)

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

trainSeqFeatures, trainSeqLabels = transformDataset(train_sents)
testSeqFeatures, testSeqLabels = transformDataset(test_sents)

vectorizer = DictVectorizer(sparse=False)
trainSeqFeatures = vectorizer.fit_transform(trainSeqFeatures)
print trainSeqFeatures.shape[1]
