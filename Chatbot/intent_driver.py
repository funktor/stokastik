import csv, intent_modeling, intent_score_predict
from sklearn.model_selection import train_test_split

questions, answers = [], []

with open(r'/Users/funktor/Downloads/tagAnswer.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        questions += [row['Question']]
        answers += [row['Answer']]

new_questions, new_answers = [], []

with open(r'/Users/funktor/Downloads/qa.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        new_questions += [row['expression_id']]
        new_answers += [row['answer']]

questions += new_questions
answers += new_answers

clusters = intent_modeling.cluster_intents(questions)

q_embeds = intent_modeling.train_doc2vec(questions)
a_embeds = intent_modeling.train_doc2vec(answers)

mydata, labels = intent_modeling.get_data_pairs(q_embeds, a_embeds, clusters)

train_data, test_data, train_labels, test_labels = train_test_split(mydata, labels, test_size=0.3, random_state=42)
scoring_model = intent_modeling.train_scoring_model(train_data, train_labels)

print(intent_score_predict.predict_scoring_model(scoring_model, test_data, test_labels))

print(intent_score_predict.kfold_cv(mydata, labels))

question = "Can I trade options using my rollover IRA account?"
answer = "You can certainly apply to trade options in your Rollover IRA."

print(intent_score_predict.predict_score_test(question, answer, q_embeds, a_embeds, scoring_model))