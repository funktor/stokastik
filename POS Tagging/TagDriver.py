import POSTagger
from nltk.corpus import brown
import nltk

brown_tagged_sents = brown.tagged_sents(categories='news')

size = int(len(brown_tagged_sents) * 0.7)

tags = [tag for (word, tag) in brown.tagged_words()]
defaultTag = nltk.FreqDist(tags).max()

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

tagsDict = {}
for index, tag in enumerate(set(tags)):
    tagsDict[tag] = index

trainSeqFeatures, trainSeqLabels = POSTagger.transformDatasetSequence(train_sents)
testSeqFeatures, testSeqLabels = POSTagger.transformDatasetSequence(test_sents)

tagProbs, startProbs, transMat, emissionMat, featuresDict = POSTagger.trainHMM(trainSeqFeatures[:30000], trainSeqLabels[:30000], tagsDict)

predictedTags = POSTagger.predictTags(testSeqFeatures[:100], tagProbs, startProbs, transMat, emissionMat, tagsDict, featuresDict)
print POSTagger.computeSeqAccuracy(predictedTags, [[tagsDict[tag] for tag in tags] for tags in testSeqLabels])

tagger = POSTagger.ngramTagger(train_sents, 2, defaultTag)
print tagger.evaluate(test_sents)

trainFeatures, trainLabels = POSTagger.transformDataset(train_sents)
testFeatures, testLabels = POSTagger.transformDataset(test_sents)

tree_model, tree_model_cv_score = POSTagger.trainDecisionTree(trainFeatures[:30000], trainLabels[:30000])
print tree_model_cv_score
print tree_model.score(testFeatures, testLabels)

nb_model, nb_model_cv_score = POSTagger.trainNaiveBayes(trainFeatures[:30000], trainLabels[:30000])
print nb_model_cv_score
print nb_model.score(testFeatures, testLabels)

nn_model, nn_model_cv_score = POSTagger.trainNN(trainFeatures[:30000], trainLabels[:30000])
print nn_model_cv_score
print nn_model.score(testFeatures, testLabels)

crf_model = POSTagger.trainCRF(trainSeqFeatures[:30000], trainSeqLabels[:30000])
pred_labels = crf_model.predict(testSeqFeatures)
print POSTagger.computeSeqAccuracy(pred_labels, testSeqLabels)