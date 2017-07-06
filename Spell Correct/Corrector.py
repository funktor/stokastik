from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import Utilities, pickle

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

def getSuggestions(clf, vectorizer, word, wordCounts, max_num_corrections=2, incorrectWordCountThreshold=10):
    possibleWords = []

    for i in range(2*len(word)):
        if i % 2 == 0:
            leftSubWord, rightSubWord = word[0:(i/2)], word[(i/2) + 1:len(word)]
        else:
            leftSubWord, rightSubWord = word[0:(i+1)/2], word[(i+1)/2:len(word)]

        subword = leftSubWord + rightSubWord
        res = test(clf, vectorizer, [subword])

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
                possibleWords = possibleWords + getSuggestions(clf, vectorizer, possibleWord, wordCounts,
                                                               max_num_corrections - 1)

    return possibleWords

def spellCorrect(clf, vectorizer, word, wordCounts, max_num_corrections=2):

    suggestions = getSuggestions(clf, vectorizer, word, wordCounts, max_num_corrections)

    suggestions = set(suggestions)

    maxSuggestedWordCount = 0
    bestSuggestedWord = word

    for suggestion in suggestions:
        if wordCounts[suggestion] > maxSuggestedWordCount:
            maxSuggestedWordCount = wordCounts[suggestion]
            bestSuggestedWord = suggestion

    return bestSuggestedWord
    
def spellSuggestCorpus(clf, vectorizer, wordCounts, vocabulary, max_num_corrections=2, incorrectWordCountThreshold=10):
    correctMapping = dict()
    
    for word, count in wordCounts.iteritems():
        if word not in vocabulary or wordCounts[word] <= incorrectWordCountThreshold:
            suggestion = spellCorrect(clf, vectorizer, word, wordCounts, max_num_corrections)
            correctMapping[word] = suggestion
            
    return correctMapping

incorrectWordCountThreshold = 10

print "Getting words..."
#contents = Utilities.getContents('train')
#words = Utilities.getWords(contents[0:1000])

contents = pickle.load(open('textdata.sav', 'rb'))
words = Utilities.getWords(contents)
wordCounts = Utilities.getWordCounts(words)

print "Getting instances..."
instances = Utilities.getInstances(words, wordCounts, min_word_count=incorrectWordCountThreshold, max_word_count=20)

vectorizer = CountVectorizer(binary=True, token_pattern='\\b[a-zA-Z]+\\b')

print "Training Neural Network..."
clf = train(instances, vectorizer)

model = dict({'classifier':clf, 'counts':wordCounts, 'vectorizer':vectorizer})

print "Saving model..."
pickle.dump(model, open('model.sav', 'wb'))

model = pickle.load(open('model.sav', 'rb'))

print "Doing correction..."
print spellCorrect(model['classifier'], model['vectorizer'], 'mertgage', model['counts'])

print "Doing correction..."
corrections = spellSuggestCorpus(model['classifier'], model['vectorizer'], model['counts'], vocabulary=set())
