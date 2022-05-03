#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import wordnet as wn 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import string
import re

# Load raw data
df = pd.read_csv('overview-of-recordings.csv')
#start cleansing
#count duplicate
duplicate=df.duplicated().sum()
Text = df[['phrase', 'prompt']]

# save English stopwords
stopwords_list = set(stopwords.words("english"))
# Clean text data
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

text = np.array(Text.loc[:,'phrase'])
new_text = []
for i in text:
    new_text.append(phrase_cleanse(i))
Text.insert(2,'new_text',new_text)
Text.to_csv(f"cleaned_data.csv", index=False)


# In[ ]:




