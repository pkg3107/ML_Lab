#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#1
def probablityAgrade():
    P_host = 0.6
    P_DS  = 0.4
    P_A_Host = 0.3
    P_A_DS = 0.2
    
    P_A = (P_A_Host * P_host) + (P_A_DS * P_DS)
    P_h_A = (P_A_Host * P_host)/P_A
    return P_h_A
result_a = probablityAgrade()
print(f"The probability that the student is a hosteler given that they scored an A grade is {result_a:.4f}")


# In[3]:


def probability_disease_given_positive_test():
    P_disease = 0.01
    P_positive_given_disease = 0.99
    specificity = 0.98
    P_positive_given_no_disease = 1 - specificity
    P_no_disease = 1 - P_disease

    P_positive = (P_positive_given_disease * P_disease) + (P_positive_given_no_disease * P_no_disease)

    P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

    return P_disease_given_positive

result_b = probability_disease_given_positive_test()
print(f"The probability of having the disease given a positive test result is {result_b:.4f}")


# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data_encoded = pd.get_dummies(data, columns=['age', 'income', 'student', 'credit_rating'])
    return data_encoded

def compute_probabilities(data):
    classes = data['buys_computer'].unique()
    priors = {}
    conditionals = {}
    
    for cls in classes:
        cls_data = data[data['buys_computer'] == cls]
        priors[cls] = len(cls_data) / len(data)
        conditionals[cls] = {}
        
        for column in data.columns:
            if column != 'buys_computer':
                counts = cls_data[column].value_counts()
                total = len(cls_data)
                conditionals[cls][column] = {}
                for value, count in counts.items():
                    conditionals[cls][column][value] = count / total

    return priors, conditionals

def predict(new_instance, priors, conditionals):
    max_prob = -1
    best_class = None

    for cls in priors:
        prob = priors[cls]
        for column, value in new_instance.items():
            if value in conditionals[cls][column]:
                prob *= conditionals[cls][column][value]
            else:
                prob *= 1e-6
        
        if prob > max_prob:
            max_prob = prob
            best_class = cls
    
    return best_class

def evaluate_accuracy(data, priors, conditionals):
    correct_predictions = 0
    total_predictions = len(data)
    
    for index, row in data.iterrows():
        actual_class = row['buys_computer']
        new_instance = row.drop('buys_computer').to_dict()
        prediction = predict(new_instance, priors, conditionals)
        
        if prediction == actual_class:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

file_path = 'q2.csv'  
data = load_and_preprocess_data(file_path)

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)


priors, conditionals = compute_probabilities(train_data)

accuracy = evaluate_accuracy(test_data, priors, conditionals)
print(f'The accuracy of the Naïve Bayes classifier is: {accuracy:.4f}')
new_instance = {
    'age_<=30': 1,
    'age_31...40': 0,
    'age_>40': 0,
    'income_high': 0,
    'income_medium': 1,
    'income_low': 0,
    'student_no': 1,
    'student_yes': 0,
    'credit_rating_excellent': 0,
    'credit_rating_fair': 1
}

prediction = predict(new_instance, priors, conditionals)
print(f'The prediction for the new instance is: {prediction}')


# In[8]:


import pandas as pd

data = {
    'Text': [
        "A great game",
        "The election was over",
        "Very clean match",
        "A clean but forgettable game",
        "It was a close election"
    ],
    'Tag': [
        "Sports",
        "Not sports",
        "Sports",
        "Sports",
        "Not sports"
    ]
}

df = pd.DataFrame(data)
file_path = 'text_tags.csv' 
df.to_csv(file_path,index=False)


# In[9]:


import pandas as pd
from collections import defaultdict
import math
import re

data = {
    'Text': [
        "A great game",
        "The election was over",
        "Very clean match",
        "A clean but forgettable game",
        "It was a close election"
    ],
    'Tag': [
        "Sports",
        "Not sports",
        "Sports",
        "Sports",
        "Not sports"
    ]
}
df = pd.DataFrame(data)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

class NaiveBayesClassifier:
    def __init__(self):
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocab = set()
        self.total_docs = 0

    def fit(self, texts, tags):
        for text, tag in zip(texts, tags):
            words = preprocess(text)
            self.total_docs += 1
            self.class_counts[tag] += 1
            for word in words:
                self.word_counts[tag][word] += 1
                self.vocab.add(word)
    
    def _class_prob(self, tag):
        return self.class_counts[tag] / self.total_docs

    def _word_prob(self, word, tag):
        word_count = self.word_counts[tag][word] + 1  
        total_words_in_class = sum(self.word_counts[tag].values()) + len(self.vocab)
        return word_count / total_words_in_class

    def predict(self, text):
        words = preprocess(text)
        scores = {}
        for tag in self.class_counts:
            score = math.log(self._class_prob(tag))
            for word in words:
                score += math.log(self._word_prob(word, tag))
            scores[tag] = score
        return max(scores, key=scores.get)

texts = df['Text']
tags = df['Tag']
classifier = NaiveBayesClassifier()
classifier.fit(texts, tags)

new_sentence = "A very close game"
predicted_tag = classifier.predict(new_sentence)
print(f'The sentence "{new_sentence}" belongs to tag: {predicted_tag}')

def evaluate_model(classifier, texts, tags):
    predictions = [classifier.predict(text) for text in texts]
    true_positives = sum((pred == true) and (true == 'Sports') for pred, true in zip(predictions, tags))
    false_positives = sum((pred != true) and (pred == 'Sports') for pred, true in zip(predictions, tags))
    false_negatives = sum((pred != true) and (true == 'Sports') for pred, true in zip(predictions, tags))
    true_negatives = sum((pred == true) and (true == 'Not sports') for pred, true in zip(predictions, tags))
    
    accuracy = (true_positives + true_negatives) / len(tags)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return accuracy, precision, recall

accuracy, precision, recall = evaluate_model(classifier, texts, tags)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')


# In[ ]:




