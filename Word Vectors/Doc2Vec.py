import Utilities
import gensim, logging
import numpy as np
from sklearn import svm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

min_count = 10
context_window = 10
vector_size = 300
downsample = 1e-5
negative_sampling = 5
num_threads = 4
num_epochs = 10

def getTrainTokens():
    trainContents = Utilities.getContents('train')
    trainTokens = Utilities.tokenizeContents(trainContents['Contents'])

    return {'Contents':trainContents['Contents'], 'Tokens':trainTokens, 'Labels':trainContents['Labels']}

def getTestTokens():
    testContents = Utilities.getContents('test')
    testTokens = Utilities.tokenizeContents(testContents['Contents'])

    return {'Contents':testContents['Contents'], 'Tokens':testTokens, 'Labels':testContents['Labels']}

def trainDoc2Vec(tokens, savePath):
    docs = [gensim.models.doc2vec.TaggedDocument(words=token, tags=['DOC_' + str(idx)])
            for idx, token in enumerate(tokens)]

    model = gensim.models.Doc2Vec(docs, min_count=min_count, window=context_window, size=vector_size,
                                  sample=downsample, negative=negative_sampling, workers=num_threads, iter=num_epochs)

    model.save(savePath)

    return model

def trainWord2Vec(tokens, savePath):
    model = gensim.models.Word2Vec(tokens, min_count=min_count, window=context_window, size=vector_size,
                                  sample=downsample, negative=negative_sampling, workers=num_threads)
    model.save(savePath)

    return model