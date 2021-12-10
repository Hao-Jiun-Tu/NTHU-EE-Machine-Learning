# -*- coding: utf-8 -*-

"""
===================== PLEASE WRITE HERE =====================
- Support Vector Machine (SVM) and Decision Tree classifier 
-  to train the 3 class of Iris plant from Iris dataset which containing 3 classes of 50 instances
-  (dataset of 4 attributes: 1: sepal length, 2: sepal width, 3: petal length, 4: petal width (all in cm) and 5: class)
- Author: Jason Hao-Jiun Tu
- Date: 2021.11.16
===================== PLEASE WRITE HERE =====================
"""

# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load data file and return two numpy arrays, including x: features and 
# y: labels
def load_data(path):
    """
    - load data and assign the features and labels
    """
    print('Loading data...')
    
    data=pd.read_csv(path, sep=",", header= None)
        
    # Print the number of samples
    n_samples = len(data)
    print('Number of samples:', n_samples)
    
    # Split the data into features and labels
    x = data.values[:, :-1].tolist()
    y = data.values[:,-1].tolist()
    
    
    return data, x, y 


# Split the data into training set and testing set
def split_dataset(x, y, testset_portion):
    print('Split dataset.')
    """
    - split the data  into a training set and a testing 
    set according to the 'testset_portion'. That is, the testing set will 
    account for 'testset_portion' of the overall data. You may use the function
    'sklearn.model_selection.train_test_split'.    
    """
    # ===================== PLEASE WRITE HERE =====================
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= testset_portion, random_state=100)
    
    # ===================== PLEASE WRITE HERE =====================
    
    return x_train, x_test, y_train, y_test    

# Train Decision tree classifier on x_train and y_train
def train_DT(x_train, y_train, depth):
    print('Start training.')
    """
    - use the function 'sklearn.DecisionTreeClassifier' to train and fit
    a classifier.    
    """    
    # ===================== PLEASE WRITE HERE =====================
    iris_clf = DecisionTreeClassifier(max_depth = depth)
    clf = iris_clf.fit(x_train, y_train)
    
    # ===================== PLEASE WRITE HERE =====================    
        
    return clf

# Train SVM classifier on x_train and y_train
def train_SVM(x_train, y_train, C, Gamma):
    print('Start training.')
    """
    - use the function 'sklearn.svm.SVC' to train and fit a classifier
    with "rbf" kernel.    
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    iris_clf = SVC(kernel = 'rbf', C = C, gamma = Gamma)
    clf = iris_clf.fit(x_train,y_train)
    
    # ===================== PLEASE WRITE HERE =====================    
        
    return clf

# Use the trained classifier to test on x_test
def test(clf, x_test):
    print('Start testing...')
    """
    - use the trained classifier to predict the classes on x_test
    
    """
    # ===================== PLEASE WRITE HERE =====================   
    
    y_pred = clf.predict(x_test)
    
    # ===================== PLEASE WRITE HERE =====================
    
    return y_pred

def plot_tree(clf, feature_names, labels):
    print('tree diagram')
    """
    - the output of decision tree is intuitive to understand and can be easily 
    visualised
    - you can use sklearn.tree.plot_tree to plot the tree diagram
    """
    # ===================== PLEASE WRITE HERE =====================
    
    tree.plot_tree(clf, feature_names = feature_names, label = labels[0]  )
    #tree.plot_tree(clf, feature_names = feature_names, label = labels[1]  )
    #tree.plot_tree(clf, feature_names = feature_names, label = labels[2]  )
    
    # ===================== PLEASE WRITE HERE =====================
    
    

def plot_decision_boundary(clf, X, y):
    """
    -plot the decision boundary surface with different Gamma and regularization
    variable(C)
    """
    # ===================== PLEASE WRITE HERE =====================
    x1s = np.linspace(0, 10, 1000)
    x2s = np.linspace(0, 5 , 1000)
    x1, x2 = np.meshgrid(x1s, x2s)
    
    plt.figure(2)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#cade7a','#538eed','#8cdba4'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.title("Decision Boundary")

    
    # ===================== PLEASE WRITE HERE =====================


# Main
if __name__=='__main__':
    # Some parameters
    path = 'iris.data'
    testset_portion = 0.2
    
    # Load data
    data, x, y = load_data(path)
    feature_names=["sepal_length","sepal_width","petal_length","petal_width"]
    labels = np.unique(np.array(y))
    
    # Encode the labels from string to integer
    lb = preprocessing.LabelEncoder()
    lb.fit(labels)
    y=lb.transform(y)
    
    # Preprocessing
    x_train, x_test, y_train, y_test = split_dataset(x, y, testset_portion)
    
    ############################################################
    ######## decision tree Classification model ################
    ## Estimate performance on unseen data
    # Set hyperparameter
    Depth=3
    
    # Train and test
    print("Training and Testing for Decision Tree:")
    clf_DT = train_DT(x_train, y_train, Depth)
    y_pred_DT = test(clf_DT, x_test)

    # get Accuracy
    acc_DT = accuracy_score(y_test, y_pred_DT)
    print('\nAccuracy Decision Tree:', round(acc_DT, 3))
    
    #get the confusion matrix
    confusion_mat_DT = confusion_matrix(y_test, y_pred_DT)
    print('\nConfusion Matrix Decision Tree:', confusion_mat_DT)
    
    ## Analysis on training behaviour
    # plot tree diagram 
    """
    to understand how the algorithm has behaved, we have to visualize the 
    splits of the decision tree
    try with different value of depth parameter
    """
    plot_tree(clf_DT, feature_names, labels)
    
    
    ############################################################
    ######## SVM Classification model ##########################
    ## Estimate performance on unseen data
    
    # Set hyperparameter
    C=1
    Gamma=10
    
    # train and test
    print("Training and Testing for SVM:")
    clf_SVM = train_SVM(x_train, y_train, C, Gamma)
    y_pred_SVM = test(clf_SVM, x_test)

    # get Accuracy
    acc_SVM = accuracy_score(y_test, y_pred_SVM)
    print('\nAccuracy SVM with gamma:{} and C:{} is'.format(Gamma, C), round(acc_SVM, 3))
    
    #get the Confusion matrix
    confusion_mat_SVM = confusion_matrix(y_test, y_pred_SVM)
    print('\nConfusion Matrix SVM with gamma:{} and C:{} is'.format(Gamma, C), confusion_mat_SVM)
     
    ## Analysis 
    # plot Decision boundary surface 
    """
    to visualize the boundaries created by 
    In decision surface plot, you can consider only 2 feature at a time
    you can consider first two feature("sepal legnth" and "sepal width") and 
    train the model
    do the analysis by changing the Gamma and C parameters of the svm.svc 
    classifier
    """
    x_train_DB=np.array(x_train)[:,:2]
    x_test_DB=np.array(x_test)[:,:2]

    clf_SVM_DB = train_SVM(x_train_DB, y_train, C, Gamma)
    plot_decision_boundary(clf_SVM_DB, x_test_DB, y_test)

    
