#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
#read cleaned data
Text = pd.read_csv('cleaned_data.csv')

# Create and fit tf_idf model
text_vectorize = TfidfVectorizer()
X_tf_idf = text_vectorize.fit_transform(Text["new_text"])

dense_list = X_tf_idf.todense().tolist()
feature_names = text_vectorize.get_feature_names()
df_tf_idf = pd.DataFrame(dense_list, columns = feature_names)

# concatenate prompt column with tf_idf matrix
text_tf_idf = pd.concat([Text["prompt"], df_tf_idf], axis = 1)
text_tf_idf.to_csv(f"tf_idf.csv", index=False)

# Create and fit hashvector model
n = Text['prompt'].nunique()
text_hashvectorize = HashingVectorizer(n_features = n*3)
X_hash = text_hashvectorize.fit_transform(Text["new_text"])

df_hash_vectorize = pd.DataFrame(X_hash.toarray())

# concatenate prompt column with hash vectorized matrix
text_hash_vectorize = pd.concat([Text["prompt"], df_hash_vectorize], axis = 1)
text_hash_vectorize.to_csv(f"hash_vectorize.csv", index=False)

# extract feature using bag_of_words
bag_word = CountVectorizer()
feature_bow = bag_word.fit_transform(Text["new_text"].values)

# maping feature 
df_bow = pd.DataFrame(feature_bow.todense().tolist(), columns = bag_word.get_feature_names())

# concatenate prompt column with bow matrix
bag_word_df = pd.concat([Text['prompt'], df_bow], axis = 1)
bag_word_df.to_csv('bag_word_df.csv',index=False)

# Create the list of list format for gensim w2v modeling 
Text['new_text_clean'] = Text['new_text'].apply(lambda x: x.split(" "))

# Train the word2vec model
w2v_model = Word2Vec(Text['new_text_clean'], min_count = 1,vector_size = 100, window = 5)


# Take the average of the word vectors for the words contained in each sentence
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

X_w2v = word_avg_vect(Text['new_text_clean'], w2v_model, 100)
# concatenate prompt column with averaged w2v matrix
df_w2v = pd.concat([Text["prompt"], X_w2v], axis = 1)
df_w2v.to_csv(f"df_w2v.csv", index=False)


# In[ ]:




