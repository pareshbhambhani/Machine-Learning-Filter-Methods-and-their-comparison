# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:10:31 2015

@author: paresh
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/paresh/anaconda/lib/python2.7/site-packages/lib/python2.7/site-packages')
from PyML import *
from sklearn import cross_validation
from sklearn import svm, feature_selection
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from copy import deepcopy
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import SelectKBest
"""
=========================== PART 1 ==========================
"""

def golub(data,labels):
    pos_examples = np.ones((1, len(data[0])))
    print pos_examples.shape
    neg_examples = np.ones((1, len(data[0])))
    print neg_examples.shape
    #Separating positive and negative examples
    for i in range (len(data)):
        if (labels[i] > 0):
            pos_examples = np.vstack((pos_examples,data[i]))
        else:
            neg_examples = np.vstack((neg_examples,data[i]))     
    print pos_examples.shape, neg_examples.shape, len(data[0]), len(data)
    pos_examples = np.delete(pos_examples, (0), axis=0)
    neg_examples = np.delete(neg_examples, (0), axis=0)    
    print pos_examples.shape, neg_examples.shape, len(data[0]), len(data)                
    mean_pos = pos_examples.mean(axis=0)
    mean_neg = neg_examples.mean(axis=0)
    std_pos = pos_examples.std(axis=0)
    std_neg = neg_examples.std(axis=0)
    print mean_pos.shape,mean_neg.shape,std_pos.shape,std_neg.shape
    scores=np.zeros(len(data[0]))
    print mean_pos
    for i in range (len(data[0])):
        std_total=std_pos[i] + std_neg[i]
        if (std_total == 0):
            std_total = 1
        scores[i] = (np.abs(mean_pos[i] - mean_neg[i]))/(std_total)
    print scores.shape
    return scores,scores    

#Merge the two files for leukemia dataset

filenames = ['/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data_leukemia/leu', '/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data_leukemia/leut']
with open('/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data_leukemia/leu_merge', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
                


#Read data and call Golub function for both datasets

X_train_arcene = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_train.data", delimiter = " ")
y_train_arcene = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_train.labels")
X_train_leukemia,y_train_leukemia = load_svmlight_file("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data_leukemia/leu_merge")
Score1_arcene,Score2_arcene=golub(X_train_arcene,y_train_arcene)
X_train_leukemia_ndarray=X_train_leukemia.toarray()
Score1_leukemia,Score2_leukemia=golub(X_train_leukemia_ndarray,y_train_leukemia)

"""
=========================== PART 2 ==========================
"""

"""
2.a
"""


SVM_L1 = svm.LinearSVC(penalty='l1',dual=False)

#Linear Classifier and feature count for arcene dataset

sum=0
for i in range (10):
    SVM_L1.fit(X_train_arcene,y_train_arcene)
    print SVM_L1.coef_[0], SVM_L1.coef_[0].shape
    coef_arcene = SVM_L1.coef_[0]
    print len(coef_arcene)
    j=0
    for i in range (len(coef_arcene)):
        if (coef_arcene[i]==0):
            j+=1
    print j, len(coef_arcene)-j
    sum+=len(coef_arcene)-j
print "Features having non-zero weight vector coefficients in arcene dataset are ---->", sum/10


#Linear Classifier and feature count for leukemia dataset

sum=0
for i in range (10):
    SVM_L1.fit(X_train_leukemia,y_train_leukemia)
    print SVM_L1.coef_[0], SVM_L1.coef_[0].shape
    coef_leukemia = SVM_L1.coef_[0]
    print len(coef_leukemia)
    j=0
    for i in range (len(coef_leukemia)):
        if (coef_leukemia[i]==0):
            j+=1
    print j, len(coef_leukemia)-j
    sum+=len(coef_leukemia)-j
print "Features having non-zero weight vector coefficients in leukemia dataset are ---->", sum/10


"""
2.b
"""
#Merge the two files for arcene dataset

filenames = ['/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_train.data', '/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_valid.data']
with open('/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_merge', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
filenames = ['/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_train.labels', '/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_valid.labels']
with open('/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_labels_merge', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
X_train_arcene_merge = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_merge", delimiter = " ")
y_train_arcene_merge = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Assignment5/data/arcene_labels_merge")

#Normalizing arcene data
X_train_arcene_norm=deepcopy(preprocessing.normalize(X_train_arcene_merge))

#Defining L2 SVM
SVM_L2 = svm.LinearSVC(penalty='l2',dual=False)

#Defining Pipeline estimator
estimators = [('svm_l1', SVM_L1), ('svm_l2', SVM_L2) ]
Pipeline_L2_L1 = Pipeline(estimators)

#Defining RFECV pipeline estimator
cv_arcene = cross_validation.StratifiedKFold(y_train_arcene_merge, 5, shuffle=True, random_state=0)
cv_leukemia = cross_validation.StratifiedKFold(y_train_leukemia, 5, shuffle=True, random_state=0)
selector = RFECV(SVM_L2, step=0.09)
rfe_svm = make_pipeline(selector, SVM_L2)


# Accuracy estimation
SVM_L1_Arcene_Accuracy = []
SVM_L1_Leukemia_Accuracy = []
SVM_L1_L2_Arcene_Accuracy = []
SVM_L1_L2_Leukemia_Accuracy = []
SVM_L2_Arcene_Accuracy = []
SVM_L2_Leukemia_Accuracy = []
SVM_RFE_Arcene_Accuracy = []
SVM_RFE_Leukemia_Accuracy = []


"""----------------------------------Part 2 L1_L2 tBD"""
for i in range (10):#needs to be changed to 10
    SVM_L1_Arcene_Accuracy.append(np.mean(cross_validation.cross_val_score(SVM_L1,X_train_arcene_norm,y_train_arcene_merge,cv=cv_arcene)))
    SVM_L1_Leukemia_Accuracy.append(np.mean(cross_validation.cross_val_score(SVM_L1,X_train_leukemia,y_train_leukemia,cv=cv_leukemia)))
    SVM_L1_L2_Arcene_Accuracy.append(np.mean(cross_validation.cross_val_score(Pipeline_L2_L1,X_train_arcene_norm,y_train_arcene_merge,cv=cv_arcene)))
    SVM_L1_L2_Leukemia_Accuracy.append(np.mean(cross_validation.cross_val_score(Pipeline_L2_L1,X_train_leukemia,y_train_leukemia,cv=cv_leukemia)))
    SVM_L2_Arcene_Accuracy.append(np.mean(cross_validation.cross_val_score(SVM_L2,X_train_arcene_norm,y_train_arcene_merge,cv=cv_arcene)))
    SVM_L2_Leukemia_Accuracy.append(np.mean(cross_validation.cross_val_score(SVM_L2,X_train_leukemia,y_train_leukemia,cv=cv_leukemia)))
    SVM_RFE_Arcene_Accuracy.append(np.mean(cross_validation.cross_val_score(rfe_svm, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))
    SVM_RFE_Leukemia_Accuracy.append(np.mean(cross_validation.cross_val_score(rfe_svm, X_train_leukemia, y_train_leukemia, cv=cv_leukemia)))

#plot accuracy for arcene dataset
x_axis_val = [1,2,3,4,5,6,7,8,9,10]
plt.figure()
plt.plot(x_axis_val,SVM_L1_Arcene_Accuracy,marker='o')
plt.plot(x_axis_val,SVM_L1_L2_Arcene_Accuracy,marker='*')
plt.plot(x_axis_val,SVM_L2_Arcene_Accuracy,marker='d')
plt.plot(x_axis_val,SVM_RFE_Arcene_Accuracy,marker='o')
plt.show()

#plot accuracy for leukemia dataset
plt.figure()
plt.plot(x_axis_val,SVM_L1_Leukemia_Accuracy,marker='o')
plt.plot(x_axis_val,SVM_L1_L2_Leukemia_Accuracy,marker='*')
plt.plot(x_axis_val,SVM_L2_Leukemia_Accuracy,marker='d')
plt.plot(x_axis_val,SVM_RFE_Leukemia_Accuracy,marker='o')
plt.show()

"""
2.c
"""
data_arcene = deepcopy(X_train_arcene_merge)
labels_arcene = deepcopy(y_train_arcene_merge)
data_leukemia = deepcopy(X_train_leukemia_ndarray)
labels_leukemia = deepcopy(y_train_leukemia)
print data_arcene.shape, y_train_arcene_merge.shape,data_leukemia.shape, y_train_leukemia.shape
k=40
w=[]
u=[]
feature_score = np.zeros(len(X_train_arcene_merge[0]))
feature_score_leukemia = np.zeros(len(X_train_leukemia_ndarray[0]))
print feature_score.shape, feature_score_leukemia
for i in range (k):
    subsample_data=[]
    subsample_labels=[]
    subsample_data_leukemia=[]
    subsample_labels_leukemia=[]
    I=[]
    J=[]
    I = np.arange(len(X_train_arcene_merge))
    J = np.arange(len(X_train_leukemia_ndarray))
    print data_arcene.shape, I.shape, data_leukemia.shape, J.shape
    np.random.shuffle(I)   
    data_arcene_shuffle=data_arcene[I]
    labels_arcene_shuffle=labels_arcene[I]
    subsample_data = data_arcene_shuffle[0:(0.8*len(data_arcene)),:]
    subsample_labels = labels_arcene_shuffle[0:(0.8*len(data_arcene))]
    np.random.shuffle(J)   
    data_leukemia_shuffle=data_leukemia[J]
    labels_leukemia_shuffle=labels_leukemia[J]
    subsample_data_leukemia = data_leukemia_shuffle[0:(0.8*len(data_leukemia)),:]
    subsample_labels_leukemia = labels_leukemia_shuffle[0:(0.8*len(data_leukemia))]
    print subsample_labels.shape, subsample_data.shape, subsample_labels_leukemia.shape, subsample_data_leukemia.shape
    SVM_L1.fit(subsample_data,subsample_labels)
    w = SVM_L1.coef_
    print w,w.shape,type(w)
    l=0
    for l in range(len(X_train_arcene_merge[0])):
        if w[0,l] !=0 :
            feature_score[l] = feature_score[l]+1
    SVM_L1.fit(subsample_data_leukemia,subsample_labels_leukemia)
    u = SVM_L1.coef_
    print u,u.shape,type(u)
    l=0
    for l in range(len(X_train_leukemia_ndarray[0])):
        if u[0,l] !=0 :
            feature_score_leukemia[l] = feature_score_leukemia[l]+1
print feature_score,feature_score_leukemia
#find the no. of features with each score between 0-40 for arcene
score_arr = np.zeros(40)
for i in range (len(X_train_arcene_merge[0])):
    for j in range (0,40):
        if feature_score[i] == j:
            score_arr[j] = score_arr[j]+1

score_arr_axis = np.zeros(40)
for i in range(40):
    score_arr_axis[i]=i
#find the no. of features with each score between 0-40 for leukemia
score_arr_leukemia = np.zeros(40)
for i in range (len(X_train_leukemia_ndarray[0])):
    for j in range (0,40):
        if feature_score_leukemia[i] == j:
            score_arr_leukemia[j] = score_arr_leukemia[j]+1

#plot score vs no of features for arcene
plt.figure()
plt.xlabel('Score')
plt.ylabel('No. of features')
plt.title('Arcene: Score vs No. of features')
plt.plot(score_arr_axis,score_arr,marker='o')
plt.yscale('log',basey=2)
plt.grid(True)
plt.show()

#plot score vs no of features for leukemia
plt.figure()
plt.xlabel('Score')
plt.ylabel('No. of features')
plt.title('Leukemia: Score vs No. of features')

plt.plot(score_arr_axis,score_arr_leukemia,marker='o')
plt.yscale('log',basey=2)
plt.grid(True)
plt.show()
"""
=========================== PART 3 ==========================
"""

def L1 (data,labels):
    k=40    
    w=[]
    feature_score = np.zeros(len(data[0]))
    for i in range (k):
        subsample_data=[]
        subsample_labels=[]
        I=[]
        I = np.arange(len(data))
        np.random.shuffle(I)
        data_shuffle=data[I]
        labels_shuffle=labels[I]
        subsample_data = data_shuffle[0:(0.8*len(data)),:]
        subsample_labels = labels_shuffle[0:(0.8*len(data))]
        SVM_L1.fit(subsample_data,subsample_labels)
        w = SVM_L1.coef_
        l=0
        for l in range(len(data[0])):
            if w[0,l] !=0 :
                feature_score[l] = feature_score[l]+1
    return feature_score,feature_score
    
#Defining L2 SVM
SVM_L2_three = svm.LinearSVC(penalty='l2',dual=False)
scale_arcene = [25,50,100,200,500,1000,2000,3000,6000,10000]
scale_leukemia = [25,50,100,200,500,1000,2000,3000,6000,7129]
Cs_arr=[0.0001,0.001,0.01,0.1,1,10,100,1000]
param_grid=[{'C':Cs_arr}]
Accuracy_arcene_golub=[]
Accuracy_arcene_RFE=[]
Accuracy_arcene_L1=[]
Accuracy_leukemia_golub=[]
Accuracy_leukemia_RFE=[]
Accuracy_leukemia_L1=[]
Grid_Search_Accuracy_arcene_golub=[]
Grid_Search_Accuracy_arcene_RFE=[]
Grid_Search_Accuracy_arcene_L1=[]
Grid_Search_Accuracy_leukemia_golub=[]
Grid_Search_Accuracy_leukemia_RFE=[]
Grid_Search_Accuracy_leukemia_L1=[]


for k in (scale_arcene):
    classifier_arcene_grid = GridSearchCV(SVM_L2_three, param_grid=param_grid) 
#Using Golub score for feature selection
    filter_selector = SelectKBest(golub, k)
    filter_svm = make_pipeline(filter_selector, SVM_L2_three)
    print 'Golub------------->'
    Accuracy_arcene_golub.append(np.mean(cross_validation.cross_val_score(filter_svm, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))
    Grid_filter_arcene_svm = make_pipeline(filter_selector, classifier_arcene_grid)
    Grid_Search_Accuracy_arcene_golub.append(np.mean(cross_validation.cross_val_score(Grid_filter_arcene_svm, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))
#Using RFE for feature selection
    RFE_selector = RFE(SVM_L2_three, step=0.09,n_features_to_select=k)
    rfe_svm = make_pipeline(RFE_selector, SVM_L2_three)
    print 'RFE------------->'
    Accuracy_arcene_RFE.append(np.mean(cross_validation.cross_val_score(rfe_svm, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))
    Grid_filter_arcene_RFE = make_pipeline(RFE_selector, classifier_arcene_grid)
    Grid_Search_Accuracy_arcene_RFE.append(np.mean(cross_validation.cross_val_score(Grid_filter_arcene_RFE, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))
#Using L1 SVM for feature selection
    L1_selector = SelectKBest(L1, k)
    L1_svm = make_pipeline(L1_selector, SVM_L2_three)
    print 'L1------------->'
    Accuracy_arcene_L1.append(np.mean(cross_validation.cross_val_score(L1_svm, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))    
    Grid_filter_arcene_L1 = make_pipeline(L1_selector, classifier_arcene_grid)
    Grid_Search_Accuracy_arcene_L1.append(np.mean(cross_validation.cross_val_score(Grid_filter_arcene_L1, X_train_arcene_norm, y_train_arcene_merge, cv=cv_arcene)))

plt.figure()
plt.plot(scale_arcene,Accuracy_arcene_golub)
plt.plot(scale_arcene,Accuracy_arcene_RFE)
plt.plot(scale_arcene,Accuracy_arcene_L1)
plt.plot(scale_arcene,Grid_Search_Accuracy_arcene_golub)
plt.plot(scale_arcene,Grid_Search_Accuracy_arcene_RFE)
plt.plot(scale_arcene,Grid_Search_Accuracy_arcene_L1)
plt.xscale('log',basex=2)
plt.xticks(scale_arcene,('25','50','100','200','500','1000','2000','3000','6000','10000'))
plt.show()

for k in (scale_leukemia):
    classifier_leukemia_grid = GridSearchCV(SVM_L2_three, param_grid=param_grid) 
#Using Golub score for feature selection
    filter_selector = SelectKBest(golub, k)
    filter_svm = make_pipeline(filter_selector, SVM_L2_three)
    print 'Golub------------->'
    Accuracy_leukemia_golub.append(np.mean(cross_validation.cross_val_score(filter_svm, X_train_leukemia_ndarray, y_train_leukemia, cv=cv_leukemia)))
    Grid_filter_leukemia_svm = make_pipeline(filter_selector, classifier_leukemia_grid)
    Grid_Search_Accuracy_leukemia_golub.append(np.mean(cross_validation.cross_val_score(Grid_filter_leukemia_svm,X_train_leukemia_ndarray, y_train_leukemia, cv=cv_leukemia)))
#Using RFE for feature selection
    RFE_selector = RFE(SVM_L2_three, step=0.09,n_features_to_select=k)
    rfe_svm = make_pipeline(RFE_selector, SVM_L2_three)
    print 'RFE------------->'
    Accuracy_leukemia_RFE.append(np.mean(cross_validation.cross_val_score(rfe_svm, X_train_leukemia, y_train_leukemia, cv=cv_leukemia)))
    Grid_filter_leukemia_RFE = make_pipeline(RFE_selector, classifier_leukemia_grid)
    Grid_Search_Accuracy_leukemia_RFE.append(np.mean(cross_validation.cross_val_score(Grid_filter_leukemia_RFE,X_train_leukemia_ndarray, y_train_leukemia, cv=cv_leukemia)))
#Using L1 SVM for feature selection
    L1_selector = SelectKBest(L1, k)
    L1_svm = make_pipeline(L1_selector, SVM_L2_three)
    print 'L1------------->'
    Accuracy_leukemia_L1.append(np.mean(cross_validation.cross_val_score(L1_svm, X_train_leukemia_ndarray, y_train_leukemia, cv=cv_leukemia)))
    Grid_filter_leukemia_L1 = make_pipeline(L1_selector, classifier_leukemia_grid)
    Grid_Search_Accuracy_leukemia_L1.append(np.mean(cross_validation.cross_val_score(Grid_filter_leukemia_L1,X_train_leukemia_ndarray, y_train_leukemia, cv=cv_leukemia)))

plt.figure()
plt.plot(scale_leukemia,Accuracy_leukemia_golub)
plt.plot(scale_leukemia,Accuracy_leukemia_RFE)
plt.plot(scale_leukemia,Accuracy_leukemia_L1)
plt.plot(scale_leukemia,Grid_Search_Accuracy_leukemia_golub)
plt.plot(scale_leukemia,Grid_Search_Accuracy_leukemia_RFE)
plt.plot(scale_leukemia,Grid_Search_Accuracy_leukemia_L1)
plt.xscale('log',basex=2)
plt.xticks(scale_leukemia,('25','50','100','200','500','1000','2000','3000','6000','7129'))
plt.show()