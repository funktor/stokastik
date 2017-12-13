import intent_modeling
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def predict_scoring_model(model, q_data, a_data, labels):
    q_data, a_data, labels = intent_modeling.transform_data_lstm(q_data, a_data, labels)
    
    print("Scoring...")
    predicted = model.predict([q_data, a_data])

    return roc_auc_score(labels, predicted, average='weighted')


def kfold_cv(q_data, a_data, labels):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    q_data, a_data, labels = np.array(q_data), np.array(a_data), np.array(labels)

    results = []

    i = 1
    for train_index, test_index in skf.split(q_data, a_data, labels):
        print("Doing CV Round ", i)

        train_q_data, train_a_data, test_q_data, test_a_data = q_data[train_index], a_data[train_index], q_data[test_index], a_data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        scoring_model = intent_modeling.train_scoring_model(train_q_data, train_a_data, train_labels)

        results.append(predict_scoring_model(scoring_model, test_q_data, test_a_data, test_labels))

        i += 1

    return results


def custom_infer_vector(embeds, tokens):
    vecs = []
    for cnt in range(50):
        vecs.append(embeds.infer_vector(tokens))

    return np.mean(vecs, axis=0)


def predict_score_train(question_idx, answer_idx, q_embeds, a_embeds, scoring_model):
    question_vector = q_embeds.docvecs[str(question_idx)]
    answer_vector = a_embeds.docvecs[str(answer_idx)]
    
    q_data = np.array([question_vector])
    a_data = np.array([answer_vector])
    labels = np.array([1])
    
    q_data, a_data, labels = intent_modeling.transform_data_lstm(q_data, a_data, labels)
    
    score = scoring_model.predict([q_data, a_data])

    return score[0][0]


def predict_score_test(question, answer, q_embeds, a_embeds, scoring_model):
    q_tokens = intent_modeling.get_tokens(question)
    a_tokens = intent_modeling.get_tokens(answer)

    question_vector = custom_infer_vector(q_embeds, q_tokens)
    answer_vector = custom_infer_vector(a_embeds, a_tokens)

    q_data = np.array([question_vector])
    a_data = np.array([answer_vector])
    labels = np.array([1])
    
    q_data, a_data, labels = intent_modeling.transform_data_lstm(q_data, a_data, labels)
    
    score = scoring_model.predict([q_data, a_data])

    return score[0][0]

def test_model_qa(question, answer, model, q_tokenizer, a_tokenizer, q_dim, a_dim):
    seq1 = q_tokenizer.texts_to_sequences([question])
    seq2 = a_tokenizer.texts_to_sequences([answer])
   
    seq1[0] = [0]*(q_dim - len(seq1[0])) + seq1[0]
    seq2[0] = [0]*(a_dim - len(seq2[0])) + seq2[0]
   
    out = model.predict([np.array(seq1), np.array(seq2)])
    print(out)
   
    pred_class = np.argmax(out[0])
   
    return pred_class