import csv, re, collections, keras, os, random
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model, model_from_json
from keras.preprocessing import text, sequence
from keras.utils import np_utils


def clean_tokens(tokens):
    return [re.sub(r'[^\w\']+', r' ', token) for token in tokens]


def tokenize(mystr):
    tokenizer = RegexpTokenizer(r'[^ ]+')
    mystr = mystr.lower()

    return tokenizer.tokenize(mystr)


def get_tokens(sentence):
    return clean_tokens(tokenize(sentence))


def get_question_intent_token(question_tokens):
    qtypes, wh_words = set(), set()

    qtypes.update(
        ["can", "could", "do", "does", "doesn't", "am", "is", "are", "should", "shouldn't", "shall", "will", "would"])

    wh_words.update(["how", "what", "what's", "why", "who", "where", "which", "when"])

    final_token, pos = "None", -1

    if question_tokens[0] in qtypes:
        pos = 0
        if len(question_tokens) > 1:
            final_token = question_tokens[0] + "__" + question_tokens[1]
        else:
            final_token = question_tokens[0]

    if final_token == "None":
        for idx in range(len(question_tokens)):
            if question_tokens[idx] in wh_words:
                pos = idx
                if idx < len(question_tokens) - 1 and question_tokens[idx + 1] in qtypes:
                    final_token = question_tokens[idx] + "__" + question_tokens[idx + 1]
                else:
                    if question_tokens[idx] == "how" and idx < len(question_tokens) - 1:
                        final_token = question_tokens[idx] + "__" + question_tokens[idx + 1]
                    else:
                        final_token = question_tokens[idx]
                break

    if final_token == "None":
        for idx in range(len(question_tokens)):
            if question_tokens[idx] in qtypes:
                pos = idx
                if idx < len(question_tokens) - 1:
                    final_token = question_tokens[idx] + "__" + question_tokens[idx + 1]
                else:
                    final_token = question_tokens[idx]
                break

    return final_token, pos


def prune_questions(questions):
    for idx in range(len(questions)):
        question = questions[idx]

        tokens = get_tokens(question)
        question_intent_token, pos = get_question_intent_token(tokens)
        
        if pos != -1:
            question = " ".join(tokens[pos:min(pos + 3, len(tokens))])
            questions[idx] = question
        
    return questions


def cluster_intents(questions):
    print("Clustering on question type...")

    intents_dict = collections.defaultdict(list)

    for idx in range(len(questions)):
        question = questions[idx]

        tokens = get_tokens(question)
        question_intent_token, pos = get_question_intent_token(tokens)

        intents_dict[question_intent_token].append(idx)

    return intents_dict


def transform_sentences(sentences):
    tokenizer = text.Tokenizer()
    sentences = [" ".join(get_tokens(sentence)) for sentence in sentences]
    tokenizer.fit_on_texts(sentences)

    return tokenizer.texts_to_sequences(sentences), tokenizer


def get_data_pairs(questions, answers, clusters):
    print("Generating training data...")

    q_data, a_data, labels = [], [], []
    all_indexes_set = set(range(len(questions)))

    questions, q_tokenizer = transform_sentences(questions)
    answers, a_tokenizer = transform_sentences(answers)
    
    questions = sequence.pad_sequences(questions)
    answers = sequence.pad_sequences(answers)

    for cluster, indexes in clusters.items():
        idx_set = set(indexes)
        negative_indices = all_indexes_set.difference(idx_set)

        for idx in indexes:
            q_data.append(questions[idx])
            a_data.append(answers[idx])
            labels.append(1)

            neg_idx = random.sample(negative_indices, 1)[0]

            q_data.append(questions[idx])
            a_data.append(answers[neg_idx])
            labels.append(0)

    return np.array(q_data), np.array(a_data), np.array(labels), q_tokenizer, a_tokenizer


def get_num_features(dataX):
    return np.amax(dataX) + 1


def train_scoring_model(q_data, a_data, labels):
    
    q_num_features, a_num_features = get_num_features(q_data), get_num_features(a_data)
    
    print("Defining architecture...")
    
    q_input = Input(shape=(q_data.shape[1], ))
    q_embedding = Embedding(output_dim=256, input_dim=q_num_features, input_length=q_data.shape[1])(q_input)
    q_lstm = LSTM(128)(q_embedding)
    
    a_input = Input(shape=(a_data.shape[1], ))
    a_embedding = Embedding(output_dim=256, input_dim=a_num_features, input_length=a_data.shape[1])(a_input)
    a_lstm = LSTM(128)(a_embedding)
    
    merged_vector = keras.layers.concatenate([q_lstm, a_lstm], axis=-1)
    
    dense_layer = Dense(64, activation='relu')(merged_vector)
    
    predictions = Dense(1, activation='sigmoid')(dense_layer)
    
    model = Model(inputs=[q_input, a_input], outputs=predictions)
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit([q_data, a_data], labels, epochs=10, batch_size=64)

    print("Scoring...")
    predicted = model.predict([q_data, a_data])
    score = roc_auc_score(labels, predicted, average="weighted")
    
    print("Score = ", score)
    
    return model