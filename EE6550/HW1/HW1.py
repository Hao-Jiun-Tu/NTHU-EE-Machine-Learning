# -*- coding: utf-8 -*-
"""
=============================================================
- EE655000 Machine Learning HW1
- Subject: Maximum A Posterior Estimation
- Dataset: https://archive.ics.uci.edu/ml/datasets/Wine
-------------------------------------------------------------
- Author: Jason Hao-Jiun Tu
- Date: 2022.03.21
=============================================================
"""

# Import libraries
import random 
import math
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SAMPLES = 178
TEST_SIZE = 54
TRAIN_SIZE = 124

def LoadData(path):
    print('Loading data...')    
    data = np.genfromtxt(path, delimiter=',')
        
    # Print the number of samples
    n_samples = len(data)
    print('Number of samples:', n_samples)
    
    return data 

def Preprocessing(data):
    # Split dataset
    print('Split dataset.')
    
    num_seq = [i for i in range(SAMPLES)]
    test_idx = random.sample(num_seq, TEST_SIZE)
    
    train = []
    test = []
    for i in range(SAMPLES):
        if i in test_idx:
            test.append(data[i])     
        else:
            train.append(data[i])
    
    # Save data
    print('Save train/test dataset.')
    train = np.array(train)
    test = np.array(test)

    pd.DataFrame(test).to_csv("./test.csv", index=False, header=False) # testing data
    pd.DataFrame(train).to_csv("./train.csv", index=False, header=False) # training data
    
    y_train, x_train = train[:, 0], train[:, 1:]
    y_test, x_test = test[:, 0], test[:, 1:]
    
    return x_train, y_train, x_test, y_test

def FeatureOrganize(x_train, y_train):
    class_num = np.zeros(3)
    feature_c1 = []
    feature_c2 = []
    feature_c3 = []
    print('Training feature organization.')
    
    # Organize features in class ascending order (class1, class2, class3)
    for i in range(y_train.size):
        class_ = int(y_train[i])   
        class_num[class_-1] += 1    
        if class_==1:
            feature_c1.append(x_train[i])
        elif class_==2:
            feature_c2.append(x_train[i])
        else:
            feature_c3.append(x_train[i])
            
    feature = [feature_c1, feature_c2, feature_c3]
    
    return feature, class_num  


def Gaussian(mean, std, x):
    p = 1/(std*math.sqrt(2*math.pi)) * math.exp(-0.5*((x-mean)/std)**2)
    return p


def MAP(feature, class_num, x_test, y_test): 
    print('Start MAP implementation.')
    priors = np.array([class_num[i]/np.sum(class_num) for i in range(3)])
    # priors = np.ones(3)*(1/3)
    # print(priors)
    
    feature_c1 = np.array(feature[0]).T
    feature_c2 = np.array(feature[1]).T
    feature_c3 = np.array(feature[2]).T
    
    mean = np.zeros((3, 13), np.float64)
    std = np.zeros((3, 13), np.float64)
    
    # Calculate mean & std of 13 features in 3 classes
    for i in range(3):
        for j in range(13):
            if i==0:
                feature_tmp = feature_c1
            elif i==1:
                feature_tmp = feature_c2
            else:
                feature_tmp = feature_c3
            
            mean[i][j] = np.average(feature_tmp[j])
            std[i][j] = np.std(feature_tmp[j])
    
    y_predict = np.ones(y_test.size)*4
    correct = 0
    for idx, data in enumerate(x_test):
        posteriors = np.ones(3, np.float64) * priors
        for class_idx in range(3):
            for feature_idx in range(13):
                likelihood = Gaussian(mean[class_idx][feature_idx], std[class_idx][feature_idx], data[feature_idx])
                posteriors[class_idx] *= likelihood

        y_predict[idx] = np.argmax(posteriors)+1
        
        if y_test[idx]==y_predict[idx]:
            correct += 1
            
    accuracy = correct/np.size(y_test)
    print('------------------------------')
    print(f'Accuracy={accuracy}')
    
    return y_predict


def PlotCurve2D(x_test, y_test, y_predict):
    # Split the label and the features from testing data
    pca = PCA(n_components=2)
    x_test_pca = pca.fit_transform(x_test)
    class_names = ['class1', 'class2', 'class3']
    labels = [1, 2, 3]
    fig = plt.figure(figsize=(12, 8))
    plt1_1 = fig.add_subplot(121)
    plt1_2 = fig.add_subplot(122)
    
    # Ground Truth
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_test==idx)[0]
        plt1_1.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], label=class_name, c=c, marker=m)

        
    plt1_1.set_title('Ground Truth')
    plt1_1.set_xlabel('PCA-feature-1')
    plt1_1.set_ylabel('PCA-feature-2')
    plt1_1.legend()
    
    # Predict
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_predict==idx)[0]
        plt1_2.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], label=class_name, c=c, marker=m)

    plt1_2.set_title('Predict')
    plt1_2.set_xlabel('PCA-feature-1')
    plt1_2.set_ylabel('PCA-feature-2')
    plt1_2.legend() 
    plt.savefig('./PCA_2D.png', dpi=300)
  
    
def PlotCurve3D(x_test, y_test, y_predict):
    # Split the label and the features from testing data
    pca = PCA(n_components=3)
    x_test_pca = pca.fit_transform(x_test)
    class_names = ['class1', 'class2', 'class3']
    labels = [1, 2, 3]
    fig = plt.figure(figsize=(12, 8))
    plt1_1 = fig.add_subplot(121, projection='3d')
    plt1_2 = fig.add_subplot(122, projection='3d')
    
    # Ground Truth
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_test==idx)[0]
        plt1_1.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], x_test_pca[class_idx, 2], label=class_name, c=c, marker=m)
            
    plt1_1.set_title('Ground Truth')
    plt1_1.set_xlabel('PCA-feature-1')
    plt1_1.set_ylabel('PCA-feature-2')
    plt1_1.legend()
    
    # Predict
    for idx, class_name, c, m in zip(labels, class_names, 'rgb', 'sxo'):
        class_idx = np.where(y_predict==idx)[0]
        plt1_2.scatter(x_test_pca[class_idx, 0], x_test_pca[class_idx, 1], x_test_pca[class_idx, 2], label=class_name, c=c, marker=m)
    
    plt1_2.set_title('Predict')
    plt1_2.set_xlabel('PCA-feature-1')
    plt1_2.set_ylabel('PCA-feature-2')
    plt1_2.set_zlabel('PCA-feature-3')
    plt1_2.legend() 
    plt.savefig('./PCA_3D.png', dpi=300)
    
if __name__=='__main__':
    # Load data
    data = LoadData('Wine.csv')

    # Preprocessing
    x_train, y_train, x_test, y_test = Preprocessing(data)
    
    # Training feature organization
    feature, class_num = FeatureOrganize(x_train, y_train)
    
    # Maximize A Posteriors to predict in testing data
    y_predict = MAP(feature, class_num, x_test, y_test)

    # Plot the curves
    PlotCurve2D(x_test, y_test, y_predict)
    PlotCurve3D(x_test, y_test, y_predict)
      