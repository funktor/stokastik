import nltk, os, eli5, collections, pickle, re
import sklearn_crfsuite, itertools
from sklearn_crfsuite import metrics
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def to_conll_iob(annotated_sentence):
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)
                        yield conll_tokens

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

    if (index > 0):
        prevWord = sentence[index - 1][0]
        prevTag = sentence[index - 1][1]
    else:
        prevWord = '<START>'
        prevTag = '<START_TAG>'

    if (index < len(sentence)-1):
        nextWord = sentence[index + 1][0]
        nextTag = sentence[index + 1][1]
    else:
        nextWord = '<END>'
        nextTag = '<END_TAG>'

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
        'prev_tag': prevTag,
        'next_tag': nextTag,
    }

    outFeatures.update(currWordShape)
    outFeatures.update(prevWordShape)
    outFeatures.update(nextWordShape)

    outFeatures.update(currWordProp)
    outFeatures.update(prevWordProp)
    outFeatures.update(nextWordProp)

    return outFeatures

def transformDataset(sentences):
    wordFeatures, wordLabels = [], []

    for sent in sentences:
        for index in range(len(sent)):
            wordFeatures.append(features(sent, index))
            wordLabels.append(sent[index][2])

    return wordFeatures, wordLabels

def get_minibatch(sentences, batch_size):
    batch = list(itertools.islice(sentences, batch_size))
    wordFeatures, wordLabels = transformDataset(batch)

    return wordFeatures, wordLabels

def iter_minibatches(sentences, batch_size):
    wordFeatures, wordLabels = get_minibatch(sentences, batch_size)

    while len(wordFeatures):
        yield wordFeatures, wordLabels
        wordFeatures, wordLabels = get_minibatch(sentences, batch_size)

def trainOnline(train_sents, tags, batch_size=500):
    minibatch_iterators = iter_minibatches(train_sents, batch_size)

    hasher = FeatureHasher(n_features=5000)

    clf = PassiveAggressiveClassifier()

    for i, (trainFeatures, trainLabels) in enumerate(minibatch_iterators):

        trainFeatures = hasher.transform(trainFeatures)
        clf.partial_fit(trainFeatures, trainLabels, tags)

        yield Pipeline([('hasher', hasher), ('classifier', clf)])

def testOnline(onlineModel, test_sents):
    testFeatures, testLabels = transformDataset(test_sents)

    for model in onlineModel:
        predLabels = model.predict(testFeatures)

        labels = list(model.classes_)
        labels.remove('O')

        print f1_score(testLabels, predLabels, average='weighted', labels=labels)

    return 1

def transformDatasetSequence(sentences):
    wordFeatures, wordLabels = [], []

    for sent in sentences:
        feats, labels = [], []

        for index in range(len(sent)):
            feats.append(features(sent, index))
            labels.append(sent[index][2])

        wordFeatures.append(feats)
        wordLabels.append(labels)

    return wordFeatures, wordLabels

def predictNERSentence(sentence, crf_model):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    annotated_sentence = [(x, y, z) for ((x, y), z) in zip(pos_tagged, ['']*len(pos_tagged))]

    testFeatures, testLabels = transformDatasetSequence([annotated_sentence])

    return crf_model.predict(testFeatures)

def trainCRF(train_sents):
    trainFeatures, trainLabels = transformDatasetSequence(train_sents)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(trainFeatures, trainLabels)

    return crf

def testCRF(crf_model, test_sents):
    testFeatures, testLabels = transformDatasetSequence(test_sents)

    predLabels = crf_model.predict(testFeatures)

    labels = list(crf_model.classes_)
    labels.remove('O')

    return metrics.flat_f1_score(testLabels, predLabels, average='weighted', labels=labels)