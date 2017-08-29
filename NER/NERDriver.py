import NER, pickle, nltk
import itertools

corpus_root = "/Users/funktor/Downloads/gmb-2.2.0"

sentences = NER.read_gmb(corpus_root)

train_sents = itertools.islice(sentences, 50000)
test_sents = itertools.islice(sentences, 5000)

crf_model = NER.trainCRF(train_sents)

pickle.dump(crf_model, open('crf_model.sav', 'wb'))

crf_model = pickle.load(open('crf_model.sav', 'rb'))

str = "Christian Bale acted as the Batman and Heath Ledger as the Joker in the movie The Dark Knight"
print NER.predictNERSentence(str, crf_model)

print NER.testCRF(crf_model, test_sents)

tags = ['O', 'B-per', 'I-per', 'B-gpe', 'I-gpe', 'B-geo', 'I-geo', 'B-org', 'I-org', 'B-tim', 'I-tim', 'B-art', 'I-art', 'B-eve', 'I-eve', 'B-nat', 'I-nat']

clf = NER.trainOnline(train_sents, tags, batch_size=500)

NER.testOnline(clf, test_sents)