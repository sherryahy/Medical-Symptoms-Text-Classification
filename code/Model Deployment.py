#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import wordnet as wn 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re


model = pickle.load(open('bestmodel.pkl', 'rb'))
pca_model = pickle.load(open('word2vec.pkl', 'rb'))
w2v_model = pickle.load(open('w2v_model.pkl', 'rb'))
stopwords_list = set(stopwords.words("english"))

def input_process(data):
    input_clean = phrase_cleanse(data)
    w2v_model = pickle.load(open('w2v_model.pkl', 'rb'))
    input_clean = [input_clean.split(" ")]
    processed_input = word_avg_vect(input_clean, w2v_model, 100)
    pca_model = pickle.load(open('word2vec.pkl', 'rb')) 
    test = pca_model.transform(processed_input)
    return test

def phrase_cleanse(phrase):
    #Tokenize and divide phrase into separate words
    token_words = word_tokenize(phrase)
    
    # Convert all texts to lower cases
    words_step1 = []
    for word_1 in token_words:
        words_step1.append(word_1.lower())
    
    #Clear all punctuation
    words_step2 = [] 
    for word_2 in words_step1:
        word_cleaned = re.sub(r'[^\w\s]','',word_2)
        words_step2.append(word_cleaned)
    
    #Clean the text list
    words_step3 = []
    for word_3 in words_step2:
        # check if every characters are alphbets
        if word_3.isalpha():
            # get rid of stop words
            if word_3 not in list(stopwords_list):
                words_step3.append(word_3)
            else:
                continue
    
    #Lemmatization - group different forms of same word which has more than 2 characters into one word
    lem = nltk.stem.WordNetLemmatizer()
    lem_list = []
    for word_4 in words_step3:
        if(len(word_4) > 2):
            lem_list.append(lem.lemmatize(word_4))
    
    join_text = " ".join(lem_list)
    
    return join_text

def word_avg_vect(data, model, num_features):
    words = set(model.wv.index_to_key)
    X_vect = np.array([np.array([model.wv[i] for i in s if i in words]) for s in data])
    X_vect_avg = []
    for v in X_vect:
        if v.size:
            X_vect_avg.append(v.mean(axis = 0))
        else:
            X_vect_avg.append(np.zeros(num_features, dtype = float))

    df_vect_avg = pd.DataFrame(X_vect_avg)
    return df_vect_avg



def predict():
    '''
    For rendering results on HTML GUI
    '''
    input =  request.form['medical diagonisis']
    final_features = input_process(input)
    prediction = model.predict(final_features)

    return output = prediction[0]

input = ()
predict(input)

