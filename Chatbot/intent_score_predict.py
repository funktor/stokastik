import intent_modeling
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def predict_scoring_model(model, test_data, test_labels):
    print("Scoring...")
    predictions = model.predict(test_data)

    return roc_auc_score(test_labels, predictions, average='weighted')


def kfold_cv(mydata, labels):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    mydata, labels = np.array(mydata), np.array(labels)

    results = []

    i = 1
    for train_index, test_index in skf.split(mydata, labels):
        print("Doing CV Round ", i)

        train_data, test_data = mydata[train_index], mydata[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        scoring_model = intent_modeling.train_scoring_model(train_data, train_labels)

        results.append(predict_scoring_model(scoring_model, test_data, test_labels))

        i += 1

    return results


def custom_infer_vector(embeds, tokens):
    vecs = []
    for cnt in range(500):
        vecs.append(embeds.infer_vector(tokens))

    return np.mean(vecs, axis=0)


def predict_score_cluster(question, answer, q_embeds, a_embeds, clusters):
    q_tokens = intent_modeling.get_tokens(question)
    a_tokens = intent_modeling.get_tokens(answer)

    question_intent_token = intent_modeling.get_question_intent_token(q_tokens)

    intent_indices = clusters[question_intent_token]

    question_vector = custom_infer_vector(q_embeds, q_tokens)

    similarities = [(intent_modeling.get_similarity(question_vector, q_embeds.docvecs[str(intent_idx)]), intent_idx)
                    for intent_idx in intent_indices if str(intent_idx) in q_embeds.docvecs]

    similarities = sorted(similarities, key=lambda k: -k[0])

    best_intent_idx = similarities[0][1]

    answer_vector = custom_infer_vector(a_embeds, a_tokens)

    return intent_modeling.get_similarity(answer_vector, a_embeds.docvecs[str(best_intent_idx)]), best_intent_idx


def predict_score_train(question_idx, answer_idx, q_embeds, a_embeds, scoring_model):
    question_vector = q_embeds.docvecs[str(question_idx)]
    answer_vector = a_embeds.docvecs[str(answer_idx)]

    test_data = [intent_modeling.get_resultant_vector(question_vector, answer_vector)]

    score = scoring_model.predict_proba(test_data)

    return score[0][1]


def predict_score_test(question, answer, q_embeds, a_embeds, scoring_model):
    q_tokens = intent_modeling.get_tokens(question)
    a_tokens = intent_modeling.get_tokens(answer)

    question_vector = custom_infer_vector(q_embeds, q_tokens)
    answer_vector = custom_infer_vector(a_embeds, a_tokens)

    test_data = [intent_modeling.get_resultant_vector(question_vector, answer_vector)]

    return scoring_model.predict_proba(test_data)[0][1]