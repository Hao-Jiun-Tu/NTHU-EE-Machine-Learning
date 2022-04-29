# -*- coding: utf-8 -*-
"""
=============================================================
- EE655000 Machine Learning HW2
-------------------------------------------------------------
- Subject: Maximum Likelihood & Bayesian Linear Regression
-------------------------------------------------------------
- Author: Jason Hao-Jiun Tu
- Date: 2022.04.24
=============================================================
"""
# Import libraries
import numpy as np
import pandas as pd
import math
import scipy.stats as st
import argparse

def BLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    ## Training Step ##
    print('------------------------------')
    print('Start BLR training ...')
    N = train_data.shape[0]
    P = O1*O2
    phi = np.zeros((N, P+2), np.float64)
    target = train_data[:,3]
    
    # Formulate Gaussian basis function
    x1 = train_data[:,0]
    x2 = train_data[:,1]
    x3 = train_data[:,2]
    bias = 1.0
    
    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x2_min = np.min(x2)
    x2_max = np.max(x2)
    s1 = (x1_max-x1_min)/(O1-1)
    s2 = (x2_max-x2_min)/(O2-1)
    mu1 = np.zeros(P, np.float64)
    mu2 = np.zeros(P, np.float64)
    for i in range(O1):
        for j in range(O2):
            k = O2*i+j
            mu1[k] = s1*i + x1_min
            mu2[k] = s2*j + x2_min
            basis_k = np.exp(-(((x1-mu1[k])**2)/(2*(s1**2)))-(((x2-mu2[k])**2)/(2*(s2**2))))
            phi[:,k] = basis_k
    phi[:,k+1] = x3
    phi[:,k+2] = np.ones(N)*bias   
    phi_t = np.transpose(phi)
    
    # Parameters for tunning
    alpha = 0.1  # W = W_ML when alpha-->0
    beta = 9
    
    # Posterior distribution over w : P(w|target, alpha, beta)
    sigma_N_inv = beta*np.matmul(phi_t, phi)
    sigma_N_inv = alpha*np.eye(sigma_N_inv.shape[0]) + sigma_N_inv
    sigma_N = np.linalg.inv(sigma_N_inv)
    mean_N = beta*np.matmul(np.matmul(sigma_N, phi_t), target)
    # Maximum posterior weight: W_MAP = m_N
    W_MAP = mean_N
    
    ## Testing Step ##
    print('Start BLR testing ...')
    test_N = test_data_feature.shape[0]
    basis = np.zeros(P+2, np.float64)
    y_BLRprediction = np.zeros(test_N, np.float64)
    for idx in range(test_N):
        test_x1 = test_data_feature[idx,0]
        test_x2 = test_data_feature[idx,1]
        test_x3 = test_data_feature[idx,2]
        
        basis[:P] = np.exp(-(((test_x1-mu1)**2)/(2*(s1**2)))-(((test_x2-mu2)**2)/(2*(s2**2))))
        basis[P] = test_x3
        basis[P+1] = bias
        
        y_BLRprediction[idx] = np.sum(W_MAP*basis)
    
    return y_BLRprediction 


def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    ## Training Step ##
    print('------------------------------')
    print('Start MLR training ...')
    N = train_data.shape[0]
    P = O1*O2
    phi = np.zeros((N, P+2), np.float64)
    target = train_data[:,3]
    
    # Formulate Gaussian basis function
    x1 = train_data[:,0]
    x2 = train_data[:,1]
    x3 = train_data[:,2]
    bias = 1.0
    
    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x2_min = np.min(x2)
    x2_max = np.max(x2)
    s1 = (x1_max-x1_min)/(O1-1)
    s2 = (x2_max-x2_min)/(O2-1)
    mu1 = np.zeros(P, np.float64)
    mu2 = np.zeros(P, np.float64)
    for i in range(O1):
        for j in range(O2):
            k = O2*i+j
            mu1[k] = s1*i + x1_min
            mu2[k] = s2*j + x2_min
            basis_k = np.exp(-(((x1-mu1[k])**2)/(2*(s1**2)))-(((x2-mu2[k])**2)/(2*(s2**2))))
            phi[:,k] = basis_k
    phi[:,P] = x3
    phi[:,P+1] = np.ones(N)*bias    
    # Solve the estimated weight for maximum likelihood 
    phi_t = np.transpose(phi)
    phi_square = np.matmul(phi_t, phi)
    phi_square_inv = np.linalg.inv(phi_square)
    W_ML = np.matmul(np.matmul(phi_square_inv, phi_t), target)
    
    ## Testing Step ##
    print('Start MLR testing ...')
    test_N = test_data_feature.shape[0]
    basis = np.zeros(P+2, np.float64)
    y_MLRprediction = np.zeros(test_N, np.float64)
    for idx in range(test_N):
        test_x1 = test_data_feature[idx,0]
        test_x2 = test_data_feature[idx,1]
        test_x3 = test_data_feature[idx,2]
        
        # Sum all (P+2) basis multiplications
        basis[:P] = np.exp(-(((test_x1-mu1)**2)/(2*(s1**2)))-(((test_x2-mu2)**2)/(2*(s2**2))))
        basis[P] = test_x3
        basis[P+1] = bias
        
        y_MLRprediction[idx] = np.sum(W_ML*basis)
    
    return y_MLRprediction 


def CalMSE(data, prediction):
    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean_squared_error = sum_squared_error/prediction.shape[0]
    return mean_squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    
    print('------------------------------')
    print('MSE of BLR = {e1}, MSE of MLR = {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()
