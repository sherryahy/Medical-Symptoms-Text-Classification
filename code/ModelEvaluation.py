#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

def matric_table(model_list, name_list,y_data, X_data):
    result = []
    for m,n,a,b in zip(model_list, name_list, y_data, X_data):
        report = []
        report.append(n)
        report.append(accuracy_score(a[0], m.predict(b[0])) * 100)
        report.append(accuracy_score(a[1], m.predict(b[1])) * 100)
        report.append(recall_score(a[1], m.predict(b[1]),average = 'weighted') * 100)
        report.append(precision_score(a[1], m.predict(b[1]),average = 'weighted') * 100)
        report.append(f1_score(a[1], m.predict(b[1]),average = 'weighted') * 100)
        result.append(report)
    df = pd.DataFrame(data = result, columns=['Model', 'Training Accuracy %', 'Testing Accuracy %','Testing precision %', 'Testing recall %', 'Testing f1_score %'])
    df = df.set_index('Model')
    return df.style.highlight_max(color = 'lightgreen', axis = 0)


# In[2]:


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


# ### Random Forest with word2vec extracted data is the best among all Random Forest Classifers

# In[3]:


rf_bow = pickle.load(open('rf_bow.pkl','rb'))
rf_tf = pickle.load(open('rf_tf.pkl','rb'))
rf_hash = pickle.load(open('rf_hash.pkl','rb'))
rf_w2v = pickle.load(open('rf_w2v.pkl','rb'))

model_list = [rf_bow,rf_tf,rf_hash,rf_w2v]
name_list = ["Random Forest with bow","Random Forest with tf_idf", "Random Forest with hash","Random Forest with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)


# ### Logistic Regression with bag-of-words extracted data is the best among all logistic Regression Classifers

# In[4]:


# print result, the warning indicate there are some type the classifers never predict. but since data is imbalence in that rare class so the accuracy won't be impacted
lr_bow = pickle.load(open('lr_bow.pkl','rb'))
lr_tf = pickle.load(open('lr_tf.pkl','rb'))
lr_hash = pickle.load(open('lr_hash.pkl','rb'))
lr_w2v = pickle.load(open('lr_w2v.pkl','rb'))

model_list = [lr_bow,lr_tf,lr_hash,lr_w2v]
name_list = ["Logistic Regression with bow","Logistic Regression with tf_idf", "Logistic Regression with hash","Logistic Regressiont with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)


# ### Support Vectors Machine with tf-idf extracted data is the best among all SVCs 

# In[5]:


svc_bow = pickle.load(open('svc_bow.pkl','rb'))
svc_tf = pickle.load(open('svc_tf.pkl','rb'))
svc_hash = pickle.load(open('svc_hash.pkl','rb'))
svc_w2v = pickle.load(open('svc_w2v.pkl','rb'))
# print result
model_list = [svc_bow,svc_tf,svc_hash,svc_w2v]
name_list = ["SVC with bow","SVC with tf_idf", "SVC with hash","SVC with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)


# ### K Nearest Neighours with bag-of-words extracted data is the best among all KNNs 

# In[6]:


knn_bow = pickle.load(open('knn_bow.pkl','rb'))
knn_tf = pickle.load(open('knn_tf.pkl','rb'))
knn_hash = pickle.load(open('knn_hash.pkl','rb'))
knn_w2v = pickle.load(open('knn_w2v.pkl','rb'))

model_list = [knn_bow,knn_tf,knn_hash,knn_w2v]
name_list = ["KNN with bow","KNN with tf_idf", "KNN with hash","KNN with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)


# ### Binomial Naive Bayes with tf_idf extracted data is the best among all BNBs 

# In[7]:


bnb_bow = pickle.load(open('bnb_bow.pkl','rb'))
bnb_tf = pickle.load(open('bnb_tf.pkl','rb'))
bnb_hash = pickle.load(open('bnb_hash.pkl','rb'))
bnb_w2v = pickle.load(open('bnb_w2v.pkl','rb'))

model_list = [bnb_bow,bnb_tf,bnb_hash,bnb_w2v]
name_list = ["Binomial Naive Bayes with bow","Binomial Naive Bayes with tf_idf", "Binomial Naive Bayes with hash","Binomial Naive Bayes with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)


# ### In conclusion, Random Forest with word2vec dataset is the winner among all classifers with highest score on test accuracy, precision, recall and F1 scores

# In[8]:


### Find the best classifer among all classifers
model_list = [rf_w2v,lr_bow,svc_tf,knn_bow,bnb_tf]
name_list = ["Random Forest with word2vec","Logistic Regression with bag-of-words", "SVC with tf_idf","KNN with bag-of-words","Binomial Naive Bayes with tf_idf"]
y_data = [[y_w2v_train,y_w2v_test], [y_bow_train,y_bow_test], [y_tf_train,y_tf_test],[y_bow_train,y_bow_test],[y_tf_train,y_tf_test]]
X_data = [[X_w2v_train,X_w2v_test], [X_bow_train,X_bow_test], [X_tf_train,X_tf_test],[X_bow_train,X_bow_test],[X_tf_train,X_tf_test]]
matric_table(model_list, name_list, y_data, X_data)


# In[ ]:




