import nltk, os, pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def getContents(type='train'):
    mydata = fetch_20newsgroups(subset=type, shuffle=True, random_state=42)

    return [" ".join(data.split("\n")) for data in mydata.data]

def myTokenizer(text):
    return nltk.regexp_tokenize(text, "\\b[a-zA-Z]{3,}\\b")

def tokenizeContents(contents):
    return [myTokenizer(content) for content in contents]

def getContexts(sentences, contextSize=5):
    targetWords, contextWords = [], []

    for sentence in sentences:
        for i in range(len(sentence)):
            leftContext = sentence[max(0, i - contextSize):max(0, i)]
            rightContext = sentence[min(len(sentence) - 1, i + 1):min(len(sentence) - 1, i + contextSize+1)]
            targetWords = targetWords + [sentence[i]]
            contextWords = contextWords + [leftContext + rightContext]

    return dict({"TargetWords":targetWords, "ContextWords":contextWords})


def breakWordIntoNgrams(word, ngram_size=2):
    if (ngram_size == 1):
        return list(word)
    else:
        ngrams = []
        for i in range(len(word) - ngram_size + 1):
            out = breakWordIntoNgrams(word[i + 1:], ngram_size - 1)
            out = [word[i] + x for x in out]
            ngrams = ngrams + out

    return ngrams


def generateInstanceFromWord(word, contextWords, min_ngram_size, max_ngram_size):
    tokens = []
    for ngram_size in range(min_ngram_size, max_ngram_size + 1):
        tokens = tokens + breakWordIntoNgrams(word, ngram_size)

    return " ".join(tokens + contextWords)


def getInstances(contexts, wordCounts, min_ngram_size=1, max_ngram_size=2, min_word_count=10, max_word_count=50):
    inputs = []
    outputs = []

    currWordCounts = dict()

    words, contextWords = contexts['TargetWords'], contexts['ContextWords']

    for i in range(len(words)):
        word, context = words[i], contextWords[i]

        if wordCounts[word] >= min_word_count and (word not in currWordCounts or currWordCounts[word] < max_word_count):
            if word not in currWordCounts:
                currWordCounts[word] = 1
            else:
                currWordCounts[word] = currWordCounts[word] + 1

            for j in range(len(word)):
                subword = word[0:j] + word[j + 1:len(word)]
                inp = generateInstanceFromWord(subword, context, min_ngram_size, max_ngram_size)
                inputs.append(inp)
                outputs.append(word[j])

    return dict({'Inputs': inputs, 'Outputs': outputs})


def getWordCounts(words):
    wordCounts = dict()

    for word in words:
        if word in wordCounts:
            wordCounts[word] = wordCounts[word] + 1
        else:
            wordCounts[word] = 1

    return wordCounts


def getWords(contents):
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=myTokenizer)
    vectorizer.fit_transform(contents)

    vocabulary = set(vectorizer.vocabulary_.keys())

    docWords = tokenizeContents(contents)

    outWords = []

    for words in docWords:
        outWords.append([word.lower() for word in words if word in vocabulary])

    return outWords


def getTokens(contents):
    tokens = tokenizeContents(contents)
    tokens = [[word.lower() for word in token] for token in tokens]

    return tokens


def readTextFiles(folderPath):
    output = []

    for root, dirs, files in os.walk(folderPath):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    contents = f.read().split("\n")
                    output = output + [" ".join(contents)]

    return output