import csv, re, gensim, collections
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import RegexpTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold


def clean_tokens(tokens):
    return [re.sub(r'[^\w\']+', r' ', token) for token in tokens]


def tokenize(mystr):
    tokenizer = RegexpTokenizer(r'[^ ]+')
    mystr = mystr.lower()

    return tokenizer.tokenize(mystr)


def get_similarity(vec_1, vec_2):
    return np.fabs(np.dot(vec_1, vec_2)) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


def get_tokens(sentence):
    return clean_tokens(tokenize(sentence))


def get_question_intent_token(question_tokens):
    qtypes, wh_words = set(), set()

    qtypes.add(
        ["can", "could", "do", "does", "doesn't", "am", "is", "are", "should", "shouldn't", "shall", "will", "would"])

    wh_words.add(["how", "what", "what's", "why", "who", "where", "which", "when"])

    final_token = "None"

    if question_tokens[0] in qtypes:
        if len(question_tokens) > 1:
            final_token = question_tokens[0] + "__" + question_tokens[1]
        else:
            final_token = question_tokens[0]

    if final_token == "None":
        for idx in range(len(question_tokens)):
            if question_tokens[idx] in wh_words:
                if idx < len(question_tokens) and question_tokens[idx + 1] in qtypes:
                    final_token = question_tokens[idx] + "__" + question_tokens[idx + 1]
                else:
                    if question_tokens[idx] == "how" and idx < len(question_tokens):
                        final_token = question_tokens[idx] + "__" + question_tokens[idx + 1]
                    else:
                        final_token = question_tokens[idx]
                break

    if final_token == "None":
        for idx in range(len(question_tokens)):
            if question_tokens[idx] in qtypes:
                if idx < len(question_tokens):
                    final_token = question_tokens[idx] + "__" + question_tokens[idx + 1]
                else:
                    final_token = question_tokens[idx]
                break

    return final_token


def cluster_intents(questions):
    print("Clustering on question type...")

    intents_dict = collections.defaultdict(list)

    for idx in range(len(questions)):
        question = questions[idx]

        tokens = get_tokens(question)
        question_intent_token = get_question_intent_token(tokens)

        intents_dict[question_intent_token].append(idx)

    return intents_dict


def get_resultant_vector(vec_1, vec_2):
    res_vec = np.array(vec_1) - np.array(vec_2)

    return res_vec ** 2


def train_doc2vec(sentences):
    taggedDocs = []

    print("Generating TaggedDocuments...")

    for idx in range(len(sentences)):
        sentence = sentences[idx]
        tokens = get_tokens(sentence)

        if len(tokens) > 0:
            taggedDocs.append(TaggedDocument(tokens, [str(idx)]))

    print("Generating embeddings...")
    model = gensim.models.Doc2Vec(alpha=0.025, size=300, window=5, min_alpha=0.025, min_count=2,
                                  workers=4, negative=5, hs=0, iter=200)

    model.build_vocab(taggedDocs)
    model.train(taggedDocs, total_examples=model.corpus_count, epochs=200)

    return model


def get_data_pairs(q_embeds, a_embeds, clusters):
    mydata, labels = [], []

    print("Generating training data...")
    for cluster, indexes in clusters.items():
        idx_set = set(indexes)

        for idx in indexes:
            if str(idx) in q_embeds.docvecs and str(idx) in a_embeds.docvecs:
                question_vec = q_embeds.docvecs[str(idx)]

                most_dissimilar = q_embeds.docvecs.most_similar(negative=[str(idx)], topn=5)

                answer_vec = a_embeds.docvecs[str(idx)]
                mydata.append(get_resultant_vector(question_vec, answer_vec))
                labels.append(1)

                for idx2, sim in most_dissimilar:
                    if int(idx2) not in idx_set and idx2 in a_embeds.docvecs:
                        answer_vec = a_embeds.docvecs[idx2]
                        mydata.append(get_resultant_vector(question_vec, answer_vec))
                        labels.append(0)

    return mydata, labels


def train_scoring_model(train_data, train_labels):
    print("Training model...")
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300), random_state=1)
    model.fit(train_data, train_labels)

    print("Scoring...")
    predictions = model.predict(train_data)

    print("AUC = ", roc_auc_score(train_labels, predictions, average='weighted'))

    return model

questions, answers = [], []

with open(r'C:\Users\a615296\Documents\tagAnswer.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        questions += [row['Question']]
        answers += [row['Answer']]

new_questions, new_answers = [], []

with open(r'C:\Users\a615296\Documents\qa.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        new_questions += [row['expression_id']]
        new_answers += [row['answer']]

questions += new_questions
answers += new_answers

clusters = cluster_intents(questions)

q_embeds = train_doc2vec(questions)
a_embeds = train_doc2vec(answers)

mydata, labels = get_data_pairs(q_embeds, a_embeds, clusters, len(questions))

train_data, test_data, train_labels, test_labels = train_test_split(mydata, labels, test_size=0.3, random_state=42)
scoring_model = train_scoring_model(train_data, train_labels)

print(predict_scoring_model(scoring_model, test_data, test_labels))

print(kfold_cv(mydata, labels))

question = "Can I trade options using my rollover IRA account?"
answer = "You can certainly apply to trade options in your Rollover IRA."

print(predict_score_test(question, answer, q_embeds, a_embeds, scoring_model))