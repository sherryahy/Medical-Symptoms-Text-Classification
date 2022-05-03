#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# fucntion for PCA as feature selection 
# set cutoff value is number of components that represents 99% of variance 
# return reduced dataset with appropriate PCA components represented 99% variance
def PCA_project(data, data_name="", threshold = 99):
    max_component = data.shape[1]
    cutoff = threshold
    covar_matrix = PCA(n_components = max_component)
    covar_matrix.fit(data)
    variance = covar_matrix.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals = 4)*100)
    index = 0
    for i in range(len(var)):
        
        if np.round(var[i]) < cutoff:
            index += 1
        else:
            break
    principal=PCA(n_components=index)
    principal.fit(data)
    print('%s reduce features from %d to %d'% (data_name,max_component, index))
    return pd.DataFrame(principal.transform(data))

#read apply PCA on our 4 dataset:bag of words, tf_idf, hash, word2vec
df_bow = pd.read_csv('bag_word_df.csv')
bow_P = PCA_project(df_bow.drop('prompt', axis=1), 'bag of words')

df_tf_idf = pd.read_csv('tf_idf.csv')
tf_idf_P= PCA_project(df_tf_idf.drop('prompt', axis=1), 'tf_idf')

df_hash_vectorize = pd.read_csv('hash_vectorize.csv')
hash_P= PCA_project(df_hash_vectorize.drop('prompt', axis=1), 'hash_vectorize')

df_w2v = pd.read_csv('df_w2v.csv')
w2v_P= PCA_project(df_w2v.drop('prompt',axis =1), 'word2vec')

#save these transformed data
bow_P.to_csv('bow_P.csv', index = False)
tf_idf_P.to_csv('tf_idf_P.csv', index = False)
hash_P.to_csv('hash_P.csv', index = False)
w2v_P.to_csv('w2v_P.csv', index = False)


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
Text = pd.read_csv('cleaned_data.csv')
y = Text["prompt"]

X_bow_train, X_bow_test, y_bow_train, y_bow_test = train_test_split(bow_P,y,test_size = 0.2, random_state =0, stratify = y)
X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split(tf_idf_P,y,test_size = 0.2, random_state =0, stratify = y)
X_hash_train, X_hash_test, y_hash_train, y_hash_test = train_test_split(hash_P,y,test_size = 0.2, random_state =0, stratify = y)
X_w2v_train, X_w2v_test, y_w2v_train, y_w2v_test = train_test_split(w2v_P,y,test_size = 0.2, random_state =0, stratify = y)

pd.DataFrame(X_bow_train).to_csv('X_bow_train.csv', index = False) 
pd.DataFrame(X_bow_test).to_csv('X_bow_test.csv', index = False)
pd.DataFrame(y_bow_train).to_csv('y_bow_train.csv', index = False) 
pd.DataFrame(y_bow_test).to_csv('y_bow_test.csv', index = False)

pd.DataFrame(X_tf_train).to_csv('X_tf_train.csv', index = False) 
pd.DataFrame(X_tf_test).to_csv('X_tf_test.csv', index = False)
pd.DataFrame(y_tf_train).to_csv('y_tf_train.csv', index = False) 
pd.DataFrame(y_tf_test).to_csv('y_tf_test.csv', index = False)

pd.DataFrame(X_hash_train).to_csv('X_hash_train.csv', index = False) 
pd.DataFrame(X_hash_test).to_csv('X_hash_test.csv', index = False)
pd.DataFrame(y_hash_train).to_csv('y_hash_train.csv', index = False) 
pd.DataFrame(y_hash_test).to_csv('y_hash_test.csv', index = False)

pd.DataFrame(X_w2v_train).to_csv('X_w2v_train.csv', index = False) 
pd.DataFrame(X_w2v_test).to_csv('X_w2v_test.csv', index = False)
pd.DataFrame(y_w2v_train).to_csv('y_w2v_train.csv', index = False) 
pd.DataFrame(y_w2v_test).to_csv('y_w2v_test.csv', index = False)


# In[ ]:




