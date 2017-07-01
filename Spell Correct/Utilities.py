import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def getContents(type='train'):
    mydata = fetch_20newsgroups(subset=type, shuffle=True, random_state=42)

    contents = [" ".join(data.split("\n")) for data in mydata.data]
    labels = mydata.target

    return {'Contents':contents, 'Labels':labels}

def myTokenizer(text):
    return nltk.regexp_tokenize(text, "\\b[a-zA-Z]{3,}\\b")

def tokenizeContents(contents):
    return [myTokenizer(content) for content in contents]

def breakWordIntoNgrams(word, ngram_size=2):
    if (ngram_size == 1):
        return list(word)
    else:
        ngrams = []
        for i in range(len(word) - ngram_size + 1):
            out = breakWordIntoNgrams(word[i+1:], ngram_size-1)
            out = [word[i] + x for x in out]
            ngrams = ngrams + out

    return ngrams

def generateInstanceFromWord(word, min_ngram_size, max_ngram_size):
    tokens = []
    for ngram_size in range(min_ngram_size, max_ngram_size + 1):
        tokens = tokens + breakWordIntoNgrams(word, ngram_size)

    return " ".join(tokens)

def getInstances(words, min_ngram_size=1, max_ngram_size=3):

    inputs = []
    outputs = []

    for word in words:
        for i in range(len(word)):
            subword = word[0:i] + word[i + 1:len(word)]
            inp = generateInstanceFromWord(subword, min_ngram_size, max_ngram_size)
            inputs.append(inp)
            outputs.append(word[i])

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

    outWords = [word for words in docWords for word in [word.lower() for word in words] if word in vocabulary]

    return outWords