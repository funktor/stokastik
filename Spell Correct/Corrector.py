from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import Utilities, editdistance, pickle

def trainNN(trainData, trainLabels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1)
    clf.fit(trainData, trainLabels)

    return clf

def train(instances, vectorizer):
    trainData = vectorizer.fit_transform(instances['Inputs'])
    trainLabels = instances['Outputs']

    return trainNN(trainData, trainLabels)

def test(clf, vectorizer, words):
    instances = [Utilities.generateInstanceFromWord(word, 1, 2) for word in words]
    testData = vectorizer.transform(instances)

    return [[x for (y, x) in sorted(zip(pred, clf.classes_), reverse=True)] for pred in clf.predict_proba(testData)]

def getSuggestions(clf, vectorizer, word, wordCounts, max_num_corrections=2):
    possibleWords = []

    for i in range(len(word)):
        subword = word[0:i] + word[i + 1:len(word)]
        res = test(clf, vectorizer, [subword])

        for j in range(3):
            char = res[0][j]
            possibleWord = word[0:i] + char + word[i + 1:len(word)]

            if (max_num_corrections == 1):
                a = possibleWord in wordCounts and word in wordCounts and wordCounts[possibleWord] > \
                                                                           wordCounts[
                                                                               word]
                b = possibleWord in wordCounts and word not in wordCounts

                if a or b:
                    possibleWords.append(possibleWord)
            else:
                possibleWords = possibleWords + getSuggestions(clf, vectorizer, possibleWord, wordCounts,
                                                               max_num_corrections - 1)

    return possibleWords

def spellCorrect(clf, vectorizer, word, wordCounts, max_num_corrections=2):

    suggestions = getSuggestions(clf, vectorizer, word, wordCounts, max_num_corrections)

    suggestions = set(suggestions)

    print suggestions

    maxSuggestedWordCount = 0
    bestSuggestedWord = word

    for suggestion in suggestions:
        if wordCounts[suggestion] > maxSuggestedWordCount:
            maxSuggestedWordCount = wordCounts[suggestion]
            bestSuggestedWord = suggestion

    return bestSuggestedWord

print "Getting words..."
contents = Utilities.getContents('train')
words = Utilities.getWords(contents['Contents'][0:1000])
wordCounts = Utilities.getWordCounts(words)

print "Getting instances..."
instances = Utilities.getInstances(words, 1, 2)

vectorizer = CountVectorizer(binary=True, token_pattern='\\b[a-zA-Z]+\\b')

print "Training Neural Network..."
clf = train(instances, vectorizer)

model = dict({'classifier':clf, 'counts':wordCounts, 'vectorizer':vectorizer})

print "Saving model..."
pickle.dump(model, open('model.sav', 'wb'))

model = pickle.load(open('model.sav', 'rb'))

print spellCorrect(model['classifier'], model['vectorizer'], 'feiescoqe', model['counts'], max_num_corrections=3)