from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import Utilities, pickle
import gensim, logging, os, editdistance

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def trainNN(trainData, trainLabels):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1)
    clf.fit(trainData, trainLabels)

    return clf


def train(instances, vectorizer):
    trainData = vectorizer.fit_transform(instances['Inputs'])
    trainLabels = instances['Outputs']

    return trainNN(trainData, trainLabels)


def trainWord2VecModel(contents):

    print "Getting tokens..."
    tokens = Utilities.getTokens(contents)

    print "Training word2vec..."
    model = gensim.models.Word2Vec(tokens, min_count=10, window=10, size=300, sample=1e-5, workers=4, hs=1, iter=10)
    model.save('word2vec')

    return model


def test(clf, vectorizer, words, contexts):
    inputs = zip(words, contexts)

    instances = [Utilities.generateInstanceFromWord(word, contextWords, 1, 2) for word, contextWords in inputs]
    testData = vectorizer.transform(instances)

    return [[x for (y, x) in sorted(zip(pred, clf.classes_), reverse=True)] for pred in clf.predict_proba(testData)]


def getSuggestions(clf, vectorizer, word, contextWords, wordCounts, max_num_corrections=2, incorrectWordCountThreshold=10):
    possibleWords = []

    for i in range(2 * len(word)+1):
        if i % 2 == 0:
            leftSubWord, rightSubWord = word[0:(i / 2)], word[(i / 2):len(word)]
        else:
            leftSubWord, rightSubWord = word[0:(i - 1) / 2], word[((i - 1) / 2) + 1:len(word)]

        subword = leftSubWord + rightSubWord
        res = test(clf, vectorizer, [subword], [contextWords])

        res[0] = [''] + res[0]

        if max_num_corrections == 1:
            corrRange = 3
        else:
            corrRange = len(res[0])

        for j in range(corrRange):
            char = res[0][j]
            possibleWord = leftSubWord + char + rightSubWord

            if (max_num_corrections == 1):
                x = possibleWord in wordCounts and wordCounts[possibleWord] > incorrectWordCountThreshold
                a = x and word in wordCounts and wordCounts[possibleWord] > wordCounts[word]
                b = x and word not in wordCounts

                if a or b:
                    possibleWords.append(possibleWord)
            else:
                possibleWords = possibleWords + getSuggestions(clf, vectorizer, possibleWord, contextWords, wordCounts,
                                                               max_num_corrections - 1)

    return possibleWords


def spellCorrect(clf, vectorizer, word, contextWords, wordCounts, max_num_corrections=2):
    suggestions = getSuggestions(clf, vectorizer, word, contextWords, wordCounts, max_num_corrections)

    suggestions = set(suggestions)

    maxSuggestedWordCount = 0
    bestSuggestedWord = word

    for suggestion in suggestions:
        if wordCounts[suggestion] > maxSuggestedWordCount:
            maxSuggestedWordCount = wordCounts[suggestion]
            bestSuggestedWord = suggestion

    return bestSuggestedWord

def spellCorrectW2V(w2vModel, word, contextWords, max_num_corrections=2):
    possibleWords = w2vModel.predict_output_word(contextWords, topn=30)

    minDist = max_num_corrections+1
    correctedWord = word

    for possibleWord, prob in possibleWords:
        dist = editdistance.eval(possibleWord, word)
        if dist < minDist and dist >= 1 and dist <= max_num_corrections and dist/len(word) <= 0.4:
            minDist = dist
            correctedWord = possibleWord

    return correctedWord


def trainNNWordModel(contents):

    print "Getting words..."
    sentences = Utilities.getWords(contents)

    print "Getting contexts..."
    contexts = Utilities.getContexts(sentences, contextSize=10)

    print "Getting word counts..."
    wordCounts = Utilities.getWordCounts(contexts['TargetWords'])

    print "Getting instances..."
    instances = Utilities.getInstances(contexts, wordCounts, min_word_count=10, max_word_count=30)

    vectorizer = CountVectorizer(binary=True, token_pattern='\\b[a-zA-Z]+\\b')

    print "Training Neural Network..."
    clf = train(instances, vectorizer)

    model = dict({'classifier': clf, 'counts': wordCounts, 'vectorizer': vectorizer})

    print "Saving model..."
    pickle.dump(model, open('model.sav', 'wb'))

    return model


trainContents = Utilities.getContents('train')
testContents = Utilities.getContents('test')

contents = trainContents + testContents

nnModel = trainNNWordModel(contents)

nnModel = pickle.load(open('model.sav', 'rb'))

print "Doing correction..."
print spellCorrect(nnModel['classifier'], nnModel['vectorizer'], 'pundue',
                   "university engineering computer network distribution".split(), nnModel['counts'])


w2vModel = trainWord2VecModel(contents)

w2vModel = gensim.models.Word2Vec.load('word2vec')

print "Doing correction..."
print spellCorrectW2V(w2vModel, "pundue", "university engineering computer network distribution".split())
