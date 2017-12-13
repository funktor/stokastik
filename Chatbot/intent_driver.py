import csv
import intent_modeling, intent_score_predict
from sklearn.model_selection import train_test_split
from importlib import reload

questions, answers = [], []

with open(r'/Users/funktor/Documents/tagAnswer.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        questions += [row['Question']]
        answers += [row['Answer']]

new_questions, new_answers = [], []

with open(r'/Users/funktor/Documents/qa.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        new_questions += [row['expression_id']]
        new_answers += [row['answer']]

questions += new_questions
answers += new_answers

clusters = intent_modeling.cluster_intents(questions)

q_embeds = intent_modeling.train_doc2vec(questions)
a_embeds = intent_modeling.train_doc2vec(answers)

q_data, a_data, labels = intent_modeling.get_data_pairs(q_embeds, a_embeds, clusters)

train_q_data, test_q_data, train_a_data, test_a_data, train_labels, test_labels = train_test_split(q_data, a_data, labels, test_size=0.3, random_state=42)

scoring_model = intent_modeling.train_scoring_model(train_q_data, train_a_data, train_labels)

intent_modeling.save_models(q_embeds, a_embeds, scoring_model)

q_embeds, a_embeds, scoring_model = intent_modeling.load_models()

print(intent_score_predict.predict_scoring_model(scoring_model, test_q_data, test_a_data, test_labels))

print(intent_score_predict.kfold_cv(q_data, a_data, labels, labels))

question = "Can I trade options using my rollover IRA account?"
answer = "no you can't"

print(intent_score_predict.predict_score_test(question, answer, q_embeds, a_embeds, scoring_model))

for idx in range(len(questions[:20])):
    print(questions[idx])
    print()
    print(answers[idx])
    print()
    print(intent_score_predict.predict_score_train(idx, idx, q_embeds, a_embeds, scoring_model))