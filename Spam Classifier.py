# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:56:56 2019

@author: Utkarsh Kant
"""

# import dependencies and dataset
import pandas as pd

# read dataset
df = pd.read_csv('...\\Spam Classifier\\SMSSpamCollection', sep='\t', names=['label', 'message'])

# print first 5 rows of dataset
df.head()

# data cleaning and pre-processing
import re  # regular expression library
import nltk  # nltk library
from nltk.corpus import stopwords  # import stopwords module to remove stopwords from dataset
from nltk.stem.porter import PorterStemmer  # import PorterStemmer module to perform stemming on dataset

ps = PorterStemmer()  # intialise PorterStemmer()
corpus = []  # to store sentences here after cleaning
    
# loop to perform cleaning (stopwords removal + stemming) on sentences
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i]) # substitute every character except a to z and A to Z with space (' ')
    review = review.lower()  # convert everything into lowercase
    review = review.split()  # create a list of words by splitting the sentence on space (' ')
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # remove stopwords then perform stemming
    review = ' '.join(review)  # re-join the processed words into a sentence
    corpus.append(review)  # append the processed sentences in the list - corpus

# creating bag of words
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000) # initialise CountVectorizer
    X = cv.fit_transform(corpus).toarray()  # apply cv to corpus and transform into array to create bag of words
# 'X' is our complete dataset of independent variables to perform modelling upon
    
# let's create the dataset of dependent variables to perform modelling upon
y = pd.get_dummies(df['label'], drop_first=True)  # convert the 'label' column from object dtype to a dummy variable

# train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# modelling using Naive Bayes Algorithm
from sklearn.naive_bayes import MultinomialNB  # classification between spam & ham with Naive Bayes

spam_detect_model = MultinomialNB().fit(X_train, y_train)  # train the model on training data
y_pred = spam_detect_model.predict(X_test)  # make predictions with the test data

# import confusion matrix for model's accuracy prediction
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)  # we have a 98% of accuracy score which is great

'''
Note - The following methods generally improve model's accuracy :
    1. Lemmatization instead of Stemming
    2. TF-IDF instead of Bag of Words
'''
