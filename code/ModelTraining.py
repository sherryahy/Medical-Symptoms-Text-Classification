#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

X_bow_train = pd.read_csv('X_bow_train.csv')
X_bow_test  = pd.read_csv('X_bow_test.csv')
y_bow_train = pd.read_csv('y_bow_train.csv')
y_bow_test  = pd.read_csv('y_bow_test.csv')


X_tf_train = pd.read_csv('X_tf_train.csv')
X_tf_test  = pd.read_csv('X_tf_test.csv')
y_tf_train = pd.read_csv('y_tf_train.csv')
y_tf_test  = pd.read_csv('y_tf_test.csv')

X_hash_train = pd.read_csv('X_hash_train.csv')
X_hash_test  = pd.read_csv('X_hash_test.csv')
y_hash_train = pd.read_csv('y_hash_train.csv')
y_hash_test  = pd.read_csv('y_hash_test.csv')

X_w2v_train = pd.read_csv('X_w2v_train.csv')
X_w2v_test  = pd.read_csv('X_w2v_test.csv')
y_w2v_train = pd.read_csv('y_w2v_train.csv')
y_w2v_test  = pd.read_csv('y_w2v_test.csv')


# In[2]:


import pickle
from sklearn.ensemble import RandomForestClassifier
# train model with all features
rf_bow = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)
rf_tf = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)
rf_hash = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)
rf_w2v = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)

rf_bow.fit(X_bow_train, y_bow_train)
rf_tf.fit(X_tf_train, y_tf_train)
rf_hash.fit(X_hash_train, y_hash_train)
rf_w2v.fit(X_w2v_train, y_w2v_train)


# In[3]:


pickle.dump(rf_bow, open('rf_bow.pkl','wb'))
pickle.dump(rf_tf, open('rf_tf.pkl','wb'))
pickle.dump(rf_hash, open('rf_hash.pkl','wb'))
pickle.dump(rf_w2v, open('rf_w2v.pkl','wb'))


# In[4]:


#train model for logistic Regression which is not inherently multiclass classifers. 
#In this case, we use  defualt auto setting that if input is binary using OVR otherwise using multnomial
from sklearn.linear_model import LogisticRegression

lr_bow = LogisticRegression()
lr_tf = LogisticRegression()
lr_hash = LogisticRegression()
lr_w2v = LogisticRegression()

lr_bow.fit(X_bow_train, y_bow_train)
lr_tf.fit(X_tf_train, y_tf_train)
lr_hash.fit(X_hash_train, y_hash_train)
lr_w2v.fit(X_w2v_train, y_w2v_train)


# In[5]:


pickle.dump(lr_bow, open('lr_bow.pkl','wb'))
pickle.dump(lr_tf, open('lr_tf.pkl','wb'))
pickle.dump(lr_hash, open('lr_hash.pkl','wb'))
pickle.dump(lr_w2v, open('lr_w2v.pkl','wb'))


# In[6]:


#train model for linear svm, which is not inherently multiclass classifers. 
#In this case, we use One VS Rest to save computing 
from sklearn.svm import SVC

svc_bow = SVC(decision_function_shape='ovr')
svc_tf = SVC(decision_function_shape='ovr')
svc_hash = SVC(decision_function_shape='ovr')
svc_w2v = SVC(decision_function_shape='ovr')

svc_bow.fit(X_bow_train, y_bow_train)
svc_tf.fit(X_tf_train, y_tf_train)
svc_hash.fit(X_hash_train, y_hash_train)
svc_w2v.fit(X_w2v_train, y_w2v_train)


# In[7]:


pickle.dump(svc_bow, open('svc_bow.pkl','wb'))
pickle.dump(svc_tf, open('svc_tf.pkl','wb'))
pickle.dump(svc_hash, open('svc_hash.pkl','wb'))
pickle.dump(svc_w2v, open('svc_w2v.pkl','wb'))


# In[8]:


#train model for KNN
from sklearn.neighbors import KNeighborsClassifier

knn_bow = KNeighborsClassifier(n_neighbors=3)
knn_tf = KNeighborsClassifier(n_neighbors=3)
knn_hash = KNeighborsClassifier(n_neighbors=3)
knn_w2v = KNeighborsClassifier(n_neighbors=3)

knn_bow.fit(X_bow_train, y_bow_train)
knn_tf.fit(X_tf_train, y_tf_train)
knn_hash.fit(X_hash_train, y_hash_train)
knn_w2v.fit(X_w2v_train, y_w2v_train)


# In[9]:


pickle.dump(knn_bow, open('knn_bow.pkl','wb'))
pickle.dump(knn_tf, open('knn_tf.pkl','wb'))
pickle.dump(knn_hash, open('knn_hash.pkl','wb'))
pickle.dump(knn_w2v, open('knn_w2v.pkl','wb'))


# In[10]:


#train model for Naive Bayes. 
#Bernoulli NB can only focus on a single keyword, 
#but will also count how many times that keyword does not occur in the document
from sklearn.naive_bayes import BernoulliNB


bnb_bow = BernoulliNB()
bnb_tf = BernoulliNB()
bnb_hash = BernoulliNB()
bnb_w2v = BernoulliNB()

bnb_bow.fit(X_bow_train, y_bow_train)
bnb_tf.fit(X_tf_train, y_tf_train)
bnb_hash.fit(X_hash_train, y_hash_train)
bnb_w2v.fit(X_w2v_train, y_w2v_train)


# In[11]:


pickle.dump(bnb_bow, open('bnb_bow.pkl','wb'))
pickle.dump(bnb_tf, open('bnb_tf.pkl','wb'))
pickle.dump(bnb_hash, open('bnb_hash.pkl','wb'))
pickle.dump(bnb_w2v, open('bnb_w2v.pkl','wb'))


# In[ ]:




