# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:43:09 2020

@author: Shubham Buchunde
"""

import nltk
import re
import pickle
from nltk.corpus import stopwords
import numpy as np
from sklearn.datasets import load_files
nltk.download("stopwords")

reviews = load_files("txt_sentoken/")
X,y = reviews.data,reviews.target

#Storing as pickle file
with open("X.pickle","wb") as f:
    pickle.dump(X,f)

with open("y.pickle","wb") as f:
    pickle.dump(y,f)

#Unpickling the file
with open("X.pickle","rb") as f:
    X = pickle.load(f)

with open("y.pickle","rb") as f:
    y = pickle.load(f)

corpus = []
for i in range(0,len(X)):
    review = re.sub(r"\W"," ",str(X[i]))
    review = review.lower()
    review = re.sub(r"\s+[a-z]\s+"," ",review)
    review = re.sub(r"^[a-z]\s+"," ",review)
    review = re.sub(r"\s+"," ",review)    
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df = 3,max_df = 0.6,stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
    
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,min_df = 3,max_df = 0.6,stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc_sc = accuracy_score(y_test,y_pred)


#Pickling the classifier
with open("classifier.pickle","wb") as f:
    pickle.dump(classifier,f)

#Pickling the vectorizer
with open("tfidfmodel.pickle","wb") as f:
    pickle.dump(vectorizer,f)


#Unpickling the classifier and the vectorizer
with open("classifier.pickle","rb") as f:
    clf = pickle.load(f)
    
with open("tfidfmodel.pickle","rb") as f:
    tfidf = pickle.load(f)
    

sample= ["You are a nice person man,have a good life"]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))











