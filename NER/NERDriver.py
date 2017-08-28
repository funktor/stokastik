import NER
import itertools

corpus_root = "/Users/funktor/Downloads/gmb-2.2.0"

sentences = NER.read_gmb(corpus_root)

train_sents = itertools.islice(sentences, 3000)
test_sents = itertools.islice(sentences, 500)

crf_model = NER.trainCRF(train_sents)
print NER.testCRF(crf_model, test_sents)

tags = ['O', 'B-per', 'I-per', 'B-gpe', 'I-gpe', 'B-geo', 'I-geo', 'B-org', 'I-org', 'B-tim', 'I-tim', 'B-art', 'I-art', 'B-eve', 'I-eve', 'B-nat', 'I-nat']

clf = NER.trainOnline(train_sents, tags, batch_size=50)

NER.testOnline(clf, test_sents)