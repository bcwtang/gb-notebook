#!/usr/bin/env python
# coding: utf-8

# In[4]:


from model import NLPModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



'''
Create the model object
The NLP model object uses a Naive Bayes classifier and a TFIDF vectorizer:
self.clf = MultinomialNB()
self.vectorizer = TfidfVectorizer()
'''

def build_model():
    model = NLPModel()
    with open ('data/train.csv') as f:
        data = pd.read_csv(f, sep=',', header=0)

    # Use only the 1 star and 5 star reviews
    # For this example, we want to only predict positive or negative sentiment using the extreme cases.
    pos_neg = data[(data['Rating']<=2) | (data['Rating']>=4)]

    ## Relabel as 0 for negative and 1 for positive¶
    pos_neg['Binary'] = pos_neg.apply(
        lambda x: 0 if x['Rating'] < 2 else 1, axis=1)

    #Fit a vectorizer to the vocabulary in the dataset
    #pos_neg.loc[:, 'Phrase']

    pos_neg.dropna(subset=['Review Text'], inplace=True)

    X = model.vectorizer_fit_transform(pos_neg.loc[:, 'Review Text'])
    print('Vectorizer fit transform complete')

    y = pos_neg.loc[:, 'Binary']

    # split X and y into training and testing sets
    # by default, it splits 75% training and 25% test
    # random_state=1 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)
    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()


if __name__ == "__main__":
    build_model()
