# -*- coding: utf-8 -*-
"""
=============================================================
- EE655000 Machine Learning HW1
-------------------------------------------------------------
- Subject: Maximum A Posterior Estimation
- Dataset: https://archive.ics.uci.edu/ml/datasets/Wine
-------------------------------------------------------------
- Author: Jason Hao-Jiun Tu
- Date: 2022.03.24
=============================================================
"""
# Import libraries
import random 
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SAMPLES = 178
TEST_SIZE = 54
TRAIN_SIZE = 124

def LoadData(path):
    print('Loading data...')    
    data = np.genfromtxt(path, delimiter=',')       
    ## Print the number of samples ##
    n_samples = len(data)
    print('Number of samples:', n_samples)
    
    return data 

def Preprocessing(data):
    print('Split dataset.')
    ## Shuffle data ## 
    data_idx = [np.where(data[:,0]==label)[0] for label in range(1, 4)]
    for class_idx in range(3):
        np.random.shuffle(data_idx[class_idx])
    
    test_size = np.ones(3).astype(np.uint8) * (TEST_SIZE//3)
    # test_size = np.array([18, 32, 4]).astype(np.uint8)
    # print(f'test_size={test_size}')
    
    ## Split dataset ##
    train = []
    test = []   
    for class_idx in range(3):
        for idx in range(len(data_idx[class_idx])):
            if idx < test_size[class_idx]: # 18 instances for each class
                test.append(data[data_idx[class_idx][idx]])
            else:
                train.append(data[data_idx[class_idx][idx]])

    ## Save data ##
    print('Save train/test dataset.')
    train = np.array(train)
    test = np.array(test)

    pd.DataFrame(test).to_csv("./test.csv", index=False, header=False)
    pd.DataFrame(train).to_csv("./train.csv", index=False, header=False)
    
    y_train, x_train = train[:, 0].astype(np.uint8), train[:, 1:]
    y_test, x_test = test[:, 0].astype(np.uint8), test[:, 1:]
    
    return x_train, y_train, x_test, y_test

def MAP(x_train, y_train, x_test, y_test): 
    print('Start MAP implementation.')
    class_num = np.array([np.where(y_train==label)[0].size for label in range(1, 4)])
    priors = np.array([class_num[i]/np.sum(class_num) for i in range(3)])
    # priors = np.ones(3)*(1/3)
    
    ## Calculate mean & std of 13 features in 3 classes ##   
    mean = np.zeros((3, 13), np.float64)
    std = np.zeros((3, 13), np.float64)
    idx = 0
    for class_idx in range(3):
        data = np.array(x_train[idx:idx+class_num[class_idx]], np.float64)
        mean[class_idx] = np.average(data, axis=0)
        std[class_idx] = np.std(data, axis=0)
        idx += class_num[class_idx]
    
    ## Calculate the posterior ##
    # ------------------------------------------------------------------------------------------------------
    # P(c|x) = P(x|c)*P(c)/P(x) (posterior ∝ likelihood*prior)
    # ------------------------------------------------------------------------------------------------------
    # likelihood = the prodoct of the probability of all features given a class (since independent features)
    #            = P(x1|c)*P(x2|c)*...*P(x13|c)
    # ------------------------------------------------------------------------------------------------------
    # posterior = (P(x1|c)*P(x2|c)*...*P(x13|c)) * P(c) 
    # ------------------------------------------------------------------------------------------------------
    y_predict = np.ones(y_test.size)*4
    correct = 0
    for idx, data in enumerate(x_test):
        posteriors = np.ones(3, np.float64) * priors
        for class_idx in range(3):
            for feature_idx in range(13):
                likelihood = st.norm(mean[class_idx][feature_idx], std[class_idx][feature_idx]).pdf(data[feature_idx])
                posteriors[class_idx] *= likelihood

        y_predict[idx] = np.argmax(posteriors)+1
        if y_test[idx]==y_predict[idx]:
            correct += 1
            
    accuracy = correct/y_test.size
    print('------------------------------')
    print(f'Accuracy={accuracy}')
    
    return y_predict

def PlotCurve2D(x_test, y_test, y_predict):
    pca = PCA(n_components=2)
    x_test_pca = pca.fit_transform(x_test)
    class_names = ['class1', 'class2', 'class3']
    labels = [1, 2, 3]
    fig = plt.figure(figsize=(12, 8))
    plt1_1 = fig.add_subplot(121)
    plt1_2 = fig.add_subplot(122)
    
    ## Ground Truth ##
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_test==idx)[0]
        plt1_1.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], label=class_name, c=c, marker=m)
  
    plt1_1.set_title('Ground Truth')
    plt1_1.set_xlabel('PCA feature1')
    plt1_1.set_ylabel('PCA feature2')
    plt1_1.legend()
    
    ## Predict ##
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_predict==idx)[0]
        plt1_2.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], label=class_name, c=c, marker=m)

    plt1_2.set_title('Predict')
    plt1_2.set_xlabel('PCA feature1')
    plt1_2.set_ylabel('PCA feature2')
    plt1_2.legend() 
    plt.savefig('./PCA_2D.png', dpi=100)
  
def PlotCurve3D(x_test, y_test, y_predict):
    pca = PCA(n_components=3)
    x_test_pca = pca.fit_transform(x_test)
    class_names = ['class1', 'class2', 'class3']
    labels = [1, 2, 3]
    fig = plt.figure(figsize=(12, 8))
    plt1_1 = fig.add_subplot(121, projection='3d')
    plt1_2 = fig.add_subplot(122, projection='3d')
    
    ## Ground Truth ##
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_test==idx)[0]
        plt1_1.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], x_test_pca[class_idx, 2], label=class_name, c=c, marker=m)
            
    plt1_1.set_title('Ground Truth')
    plt1_1.set_xlabel('PCA feature1')
    plt1_1.set_ylabel('PCA feature2')
    plt1_1.set_zlabel('PCA feature3')
    plt1_1.legend()
    
    ## Predict ##
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_predict==idx)[0]
        plt1_2.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], x_test_pca[class_idx, 2], label=class_name, c=c, marker=m)
    
    plt1_2.set_title('Predict')
    plt1_2.set_xlabel('PCA feature1')
    plt1_2.set_ylabel('PCA feature2')
    plt1_2.set_zlabel('PCA feature3')
    plt1_2.legend() 
    plt.savefig('./PCA_3D.png', dpi=100)  
    
if __name__=='__main__':
    ## Load data ##
    data = LoadData('Wine.csv')
    ## Preprocessing (Split dataset into train/test and Save) ##
    x_train, y_train, x_test, y_test = Preprocessing(data)
    ## Maximize A Posteriors to predict in testing data ##
    y_predict = MAP(x_train, y_train, x_test, y_test)
    ## Plot the curves ##
    PlotCurve2D(x_test, y_test, y_predict)
    PlotCurve3D(x_test, y_test, y_predict)
    