#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 02:28:47 2017

@author: 
Fordham University    
CISC6930 - Data Mining
Runtime: 1 mins
"""

import math
import numpy as np
import pandas as pd
import timeit
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

unknown = ' ?'
headers = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
               "occupation", "relationship", "race", "sex", "capital_gain", 
               "capital_loss", "hours_per_week", "native_country", "label"]

# use most correlated columns with missing value column to impute
L1 = [['age','sex'],'workclass'] 
L2 = [['education_num','race'], 'native_country']
L3 = [['education_num','sex'], 'occupation']

# use all known columns to impute missing values
known = ["age", "fnlwgt", "education_num", "marital_status", "relationship", "race", "sex", "capital_gain", 
               "capital_loss", "hours_per_week"]

LL1 = [known,'workclass'] 
LL2 = [known, 'native_country']
LL3 = [known, 'occupation']

# set print precison to 2 decimals
np.set_printoptions(precision=2)

def readCSV(dataset):
    #read dataset from csv file        
    df = pd.read_csv(dataset, names = headers)
    return df

#print statistics
def statistics(df, headers):
    stats = dict()
    for i in range(len(headers)):
        stats.update({headers[i]: set(df.iloc[:,i])})
    return stats

# find missing value columns
def find_unknown(df, headers, unknown):
    missing = 0
    for header in headers:        
        yn = (unknown in list((df[header]).values))
        if yn == True:
            missing += 1
            print("There are missing data in column", header)            
    if missing == 0:
        print("There is no missing data.")

def unknown_list(df, header):
    val = df[header].values
    unknown_lst = np.where(val == unknown)[0]
    return unknown_lst

# use mode to impute columns having categorical values
def impute_mode(df, df_test):
    unknown_to_NAN(df, df_test)
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    df_test = df_test.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return df, df_test

# use predition model to impute missing data
def impute_by_model(df, df_test, impList, classifier):
    
    # convert ' ?' to NAN sothat those values will be converted to -1 when tranform to numerical
    df, df_test = unknown_to_NAN(df, df_test)   
    
    # create a new df by dropping all rows having NAN values 
    # only to build model for imputation 
    dropna_df = df.dropna(how='any').reset_index(drop=True)
    
    # before convert both df, df_test to numerical, replace below value to its column mode
    # so that, native_country will have same numerical value for each country
    df.__getitem__('native_country').replace(' Holand-Netherlands',' United-States')
    
    # convert to numerical
    num_dropna_df = df2num(dropna_df,headers)    
    num_df_test = df2num(df_test,headers)    
    num_df=df2num(df,headers)
    
    # to learn model on dataset which dropped rows contains missing values
    Xtr_train = num_dropna_df[impList[0]].values
    ytr_train = num_dropna_df[impList[1]].values
    
    
    # colum missing value from training data, use to impute training set
    Xtr_test = num_df[impList[0]].values
    
    # colum missing value from test data, use to impute training set
    Xt_test = num_df_test[impList[0]].values
    
    clf = BalancedBaggingClassifier(base_estimator=classifier,
                                    ratio='auto', 
                                    random_state=0)
    clf.fit(Xtr_train, ytr_train)
    
    # impute training data
    ytr_pred = clf.predict(Xtr_test)    
    lst = df.loc[num_df[impList[1]]==-1,impList[1]].index.tolist()
    num_df.loc[lst,impList[1]]=ytr_pred[lst]
    
    
    # impute test data
    yt_pred = clf.predict(Xt_test)    
    lstt = df_test.loc[num_df_test[impList[1]]==-1,impList[1]].index.tolist()
    num_df_test.loc[lstt,impList[1]]=yt_pred[lstt]    
    # return df, df_test
    return df, df_test

def model_imputation(df, df_test, classifier):
    # impute missing values
    df, df_test = impute_by_model(df, df_test, L1, classifier)
    df, df_test = impute_by_model(df, df_test, L2, classifier)
    df, df_test = impute_by_model(df, df_test, L3, classifier)
    return df, df_test

# convert all unknown values to NAN so that they will be converted to -1
# when transformed to correspondent numerical values
def unknown_to_NAN(df, df_test):
    df=df.replace('[?]',np.NAN,regex=True)
    df_test=df_test.replace('[?]',np.NAN,regex=True)
    return df, df_test

# convert string values to numerical values
def df2num(df, headers):
    for i in range(len(headers)-1):
        cat_columns = df.select_dtypes(['object']).columns
        
    for col in cat_columns:
        df[col] = df[col].astype('category')
        
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

# convert datafram to numpy
def df2numpy(df):
    X = df.iloc[:,:len(df.loc[0])-1].values
    y = df["label"].values
    return X,y

# prepare to binning
def round_up(x, base):
    return (base * np.round(x/base)).astype(int)

# categorize continous values
def binning(X):
    X[:,0] = round_up(X[:,0],base=5)/5
    X[:,2] = round_up(X[:,2],base=100000)/100000    
    X[:,9] = round_up(X[:,9], base = 10000)/10000
    X[:,10]= round_up(X[:,10], base = 1000)/1000        
    return X

# minmax normalization
def minmax(X, Xt):
    minx = X.min(axis=0)
    trX = (X - minx) / (X.max(axis=0) - minx)
    testX = (Xt - minx) / (X.max(axis=0) - minx)
    return trX, testX

# Encode categorical integer features using a one-hot aka one-of-K scheme
def categ_encoder(X_train, X_test):
    enc = OneHotEncoder()
    enc.fit(X_train)
    X_tr = enc.transform(X_train).toarray()
    X_ts = enc.transform(X_test).toarray()
    return X_tr, X_ts

# z-score normalization
def normalize(X_train, X_test):
    normX_train = np.zeros((len(X_train),len(X_train[0])))
    normX_test = np.zeros((len(X_test),len(X_test[0])))
    for col in range(len(X_train[0])):
        m = np.mean(X_train[:,col], dtype=np.float64)
        stdv = np.std(X_train[:,col], dtype=np.float64)
        if (stdv > 0.01) :
            normX_train[:, col] = (X_train[:,col]-m)/stdv
            normX_test[:,col] = (X_test[:,col]-m)/stdv
    return normX_train, normX_test

# person correlation
def pearson(x,y):
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    N = len(y)
    for i in range(N):
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]
    mean_x = mean_x / N
    mean_y = mean_y / N
    pop_sd_x = math.sqrt((sum_sq_x/N) - (mean_x * mean_x))
    pop_sd_y = math.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    return correlation

# select columns for wrapper
def selected_features_data(X, sorted_feat, m):
    selected_feats_X = X[:, sorted_feat[0]]
    if m > 1:
        for i in range(1,m):
            selected_feats_X = np.vstack((selected_feats_X, (X[:, sorted_feat[i]])))
    else:
        selected_feats_X = np.vstack((selected_feats_X, np.zeros(len(X))))
    return selected_feats_X.T

# selection feature criterion for warpper           
def clf_wrapper(classifier, X_train, y_train, X_test, y_test):
    clf = BalancedBaggingClassifier(base_estimator=classifier,
                                    ratio='auto', 
                                    replacement=False, 
                                    random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cfm = confusion_matrix(y_test, y_pred)
    
    # Predictive Value
    PPV = cfm[0,0]/(cfm[0,0]+cfm[0,1])
    NPV = cfm[1,1]/(cfm[1,0]+cfm[1,1])
    ACR = (cfm[0,0]+cfm[1,1])/(cfm[0,0]+cfm[1,1]+cfm[1,0]+cfm[0,1])
    return (PPV+NPV+ACR)/3

# selection feature wrapper          
def wrapper_method(classifier,X,y):
    # X - nX_train, y = y_train
    feat_dict = {}   # empty set for selected features
    print(feat_dict)
    improve = True
    prev_acr = -1
    ratio = 0.8
    divide_row = round(ratio * len(y))
    
    while improve:
        acr = np.zeros(len(X[0]))
        for feat in range(0,len(X[0])):
            sltd_lst = list(feat_dict.keys())
            sltd_lst.append(feat)
            sltd_x = selected_features_data(X, sltd_lst, len(sltd_lst))
            X_train, y_train = sltd_x[0:divide_row], y[0:divide_row]
            X_test, y_test = sltd_x[divide_row+1:], y[divide_row+1:]
            # implement model
            acr[feat] = clf_wrapper(classifier, X_train, y_train, X_test, y_test)                        
            ###
        if len(feat_dict) > 0:
            for i in feat_dict.keys():
                acr[i] = -1              # already selected feature
        idx_max = np.argmax(acr)
        if acr[idx_max] > prev_acr:
            feat_dict[idx_max] = format(acr[idx_max],'0.3f')
            print(feat_dict)
        else:        
            improve = False
        prev_acr = acr[idx_max]

# balanced learn with class-weight must be assigned
def blearn(classifier, X_train, y_train, X_test, y_test):
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    printStats(y_test, y_pred)
    return clf, y_pred

# imbalanced learn with combination of bagging and undersampling
def imblearn_(classifier, X_train, y_train, X_test, y_test):
    clf = BalancedBaggingClassifier(base_estimator=classifier,
                                    ratio='auto', 
                                    random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    printStats(y_test, y_pred)
    return clf, y_pred

# adaBoost ensemble method
def adaBoost(classifier, X_train, y_train, X_test, y_test):
    clf = AdaBoostClassifier(base_estimator=classifier, algorithm='SAMME',random_state =0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #y_pred[g10k]=1
    printStats(y_test, y_pred)
    return clf, y_pred

# majority vote from several classifiers
def ensemble_(y_pred_list):
    y_pred_mat = np.zeros((len(y_pred_list),len(y_pred_list[0])))
    for i in range(0, len(y_pred_list)):
        y_pred_mat[i] = y_pred_list[i]
    y_pred_fin = np.around(np.mean(y_pred_mat,0))
    return y_pred_fin

# print confusion matrix, preditive accuracy
def printStats(y_true, y_pred):
    print('Test set performance:')
    print()
    print(' _Confusion matrix:')
    cfm = confusion_matrix(y_true, y_pred)
    print(cfm/len(y_true))
    print()
    print('_Prediction accuracy for instances labeled <=50K is: %0.2f'% (cfm[0,0]/(cfm[0,0]+cfm[0,1])))
    print()
    print('_Prediction accuracy for instances labeled >50K is: %0.2f' % (cfm[1,1]/(cfm[1,0]+cfm[1,1])) )
    print()
    print('_Overall Testset Accuracy:')
    print('%0.2f' % accuracy_score(y_true, y_pred))   

# imputation: 'use_model', 'use_mode', 'unchange'
def process_data(imputation):
    df = readCSV('census-income.data.csv')
    df_test = readCSV('census-income.test.csv')
    
    # Data imputation
    # Replace Holand-Netherlands with United-States so that when convert training and 
    # test datas into numerical, they both have same numbers corresponding with each country
    # Holand-Netherlands is located at df["native_country"][19609]
    df.__getitem__('native_country').replace(' Holand-Netherlands',' United-States')
    
    # Imputation
    if imputation == 'use_model':
        df, df_test = model_imputation(df, df_test, DecisionTreeClassifier())
    elif imputation == 'use_mode':
        df, df_test = impute_mode(df, df_test)
        # Conversion to numerical data
        df2num(df, headers)
        df2num(df_test, headers)
    elif imputation == 'unchange':
        df, df_test = unknown_to_NAN(df, df_test)
        # Conversion to numerical data
        df2num(df, headers)
        df2num(df_test, headers)
    
    # Conversion of dataframes to numpy arrays
    X_train, y_train = df2numpy(df)
    X_train[19609, len(X_train[0])-1] = 40
    X_test, y_test = df2numpy(df_test)
    
    #g10k = df_test.loc[df_test['capital_gain']>10000,'capital_gain'].index.tolist()
    
    bX_train = binning(X_train)
    bX_test = binning(X_test)
    
    X_tr, X_ts = categ_encoder(bX_train, bX_test)
    
    # Drop Education column because it has same values as education-num column
    X_train = np.delete(X_train,3,1)
    X_test = np.delete(X_test,3,1)
 
    # Data normalization
    nX_train, nX_test = normalize(X_train, X_test)
    return X_tr, X_ts

def transform(df, df_test):
    # Conversion of dataframes to numpy arrays
    X_train, y_train = df2numpy(df)
    X_train[19609, len(X_train[0])-1] = 40
    X_test, y_test = df2numpy(df_test)
    
    # Drop Education column because it has same values as education-num column
    X_train = np.delete(X_train,3,1)
    X_test = np.delete(X_test,3,1)
    return X_train, y_train, X_test, y_test

def main(imputation):
    start = timeit.default_timer()
    
    df = readCSV('census-income.data.csv')
    df_test = readCSV('census-income.test.csv')
    
    #g10k = df_test.loc[df_test['capital_gain']>10000,'label'].index.tolist()
    
    # Data imputation
    # Replace Holand-Netherlands with United-States so that when convert training and 
    # test datas into numerical, they both have same numbers corresponding with each country
    # Holand-Netherlands is located at df["native_country"][19609]
    df.__getitem__('native_country').replace(' Holand-Netherlands',' United-States')
    

    # Imputation by prediction model or mode or keep unchange, then convert to numerical
    if imputation == 'use_model':
        start1 = timeit.default_timer()
        df, df_test = model_imputation(df, df_test, DecisionTreeClassifier())
        stop1 = timeit.default_timer()
        print("Runtime of imputation: ", format(stop1-start1, "0.0f"), " sec")
    elif imputation == 'use_mode':
        df, df_test = impute_mode(df, df_test)
        # Conversion to numerical data
        df2num(df, headers)
        df2num(df_test, headers)
    elif imputation == 'unchange':
        df, df_test = unknown_to_NAN(df, df_test)
        # Conversion to numerical data
        df2num(df, headers)
        df2num(df_test, headers)
    
    # Convert to numpy 
    X_train, y_train, X_test, y_test = transform(df, df_test)
    
    # Data normalization
    nX_train, nX_test = normalize(X_train, X_test)            
    
    # Call different classifiers
    
    print('-----Bagging Decision tree-------')    
    clf1, y_pred1 = imblearn_(DecisionTreeClassifier(criterion='entropy'), nX_train, y_train, nX_test, y_test)
    print()
    
    print('-----AdaBoost Decision Tree------')
    clf2, y_pred2 = imblearn_(AdaBoostClassifier(), X_train, y_train, X_test, y_test)
    print()
    
    print('-----k-Neighbor(3)---------------')
    important_feats = [0,3,4,5,6,8,9,10,11]
    clf3, y_pred3 = imblearn_(KNeighborsClassifier(n_neighbors=3), nX_train[:,important_feats], y_train, 
                              nX_test[:,important_feats], y_test)
    print()
    
    print('--------SVM----------------------')
    clf4, y_pred4 = blearn(SVC(C=1, class_weight = {0:0.3, 1:0.7}), nX_train, y_train, nX_test, y_test)
    print()
    
    print('-------Random Forest-------------')
    clf5, y_pred5 = imblearn_(RandomForestClassifier(criterion='entropy'), nX_train, y_train, nX_test, y_test)
    print()
 
    X_tr, X_ts= process_data('use_mode')
    
    print('-----Naive Bayes-----------------')
    clf6, y_pred6 = imblearn_(BernoulliNB(),X_tr,  y_train, X_ts, y_test)
    
    print()
       
    print('-----Logistic Regression---------')
    clf7, y_pred7 = imblearn_(LogisticRegression(max_iter=1000, class_weight={0:0.47, 1:0.53}),X_tr,  y_train, X_ts, y_test)
    print()
        
    # ENSEMBLE ALL CLASSIFIERS
    print('-----------------------------------------')
    print('------Majority Vote For 7 Learners-------')
    
    # combine all y_pred together than take majority vote
    y_pred_list = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7]
    y_pred = ensemble_(y_pred_list)
    
    # print prediction confusion matrix on test data 
    printStats(y_test, y_pred)
    
    stop = timeit.default_timer()
    print("Runtime: ", format(stop-start, "0.0f"), " sec")
     
main('use_model')
