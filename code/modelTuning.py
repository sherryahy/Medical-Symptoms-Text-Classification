#!/usr/bin/env python
# coding: utf-8

# ### In conclusion, KNN with bage of words dataset is the winner among all classifers with highest score on test accuracy, precision, recall and F1 scores, The Random Forest is best on the Training accuracy. 

# In[ ]:


### Find the best classifer among all classifers
model_list = [rf_w2v,lr_bow,svc_tf,knn_bow,gnb_tf]
name_list = ["Random Forest with w2v","Logistic Regression with bow", "SVC with tf","KNN withbow","gaussian Naive Bayes with tf"]
y_data = [[y_w2v_train,y_w2v_test], [y_bow_train,y_bow_test], [y_tf_train,y_tf_test],[y_bow_train,y_bow_test],[y_tf_train,y_tf_test]]
X_data = [[X_w2v_train,X_w2v_test], [X_bow_train,X_bow_test], [X_tf_train,X_tf_test],[X_bow_train,X_bow_test],[X_tf_train,X_tf_test]]
matric_table(model_list, name_list, y_data, X_data)


# In[ ]:


##Further tuning the KNN  classifer with bow
best_models = []
n_neighbors = [3,5,7,9]
weights = ['uniform','distance']
ps = [1,2]

def KNN_clf(n_neighbors, weight, p):
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weight,p = p)
    knn.fit(X_bow_train, y_bow_train)
    y_pred = knn.predict(X_bow_test)
    n = accuracy_score(y_bow_test,y_pred)
    best_models.append((n_neighbors, weight, p ,n))

for c in n_neighbors:
    for w in weights:
        for p in ps:
            KNN_clf(c, w, p)

print(max(best_models,key=lambda item:item[3]))


# In[ ]:


##Further tuning the random forest classifer with w2v
best_models = []
crit = ['gini', 'entropy']
max_d = range(1,20,4)
min_s_leaf = range(1,20,4)
n_est = [50, 100, 200]

def RF_clf(crit, max_d, min_s_leaf, n_est):
    forest = RandomForestClassifier(criterion=crit, max_depth=max_d, min_samples_leaf=min_s_leaf, n_estimators=n_est, random_state=1)
    forest.fit(X_w2v_train, y_w2v_train)
    y_pred = forest.predict(X_w2v_test)
    n = accuracy_score(y_w2v_test,y_pred)
    best_models.append((crit,max_d,min_s_leaf,n_est,n))


for c in crit:
    for md in max_d:
        for msl in min_s_leaf:
            for n_e in n_est:
                RF_clf(c, md, msl, n_e)


# In[ ]:


Knn_best = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform' ,p = 1)
Rf_best = RandomForestClassifier(criterion='gini', max_depth=13, min_samples_leaf=1, n_estimators=50, random_state=1)
Knn_best.fit(X_bow_train, y_bow_train)
Rf_best.fit(X_w2v_train, y_w2v_train)

model_list = [Knn_best, Rf_best]
name_list = ["Tuned KNN", 'Tuned Randome Forest']
y_data = [[y_bow_train,y_bow_test], [y_w2v_train,y_w2v_test] ]
X_data = [[X_bow_train,X_bow_test], [X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)


# In[ ]:


# too perfect, maybe we don't need graph just use our table to explain
import scikitplot as skplt
import matplotlib.pyplot as plt


skplt.metrics.plot_roc(y_w2v_test, Rf_best.predict_proba(X_w2v_test)
                                          ,text_fontsize = 'small'
                                          ,title = ' ROC for best model'
                                          ,figsize = (12,8))
plt.show()


# In[ ]:


skplt.metrics.plot_precision_recall_curve(y_w2v_test, Rf_best.predict_proba(X_w2v_test),
                                          text_fontsize = 'small'
                                          ,title = 'PR Curve for best model'
                                          ,figsize = (12,8))
plt.show()

