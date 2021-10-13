"""
===================== PLEASE WRITE HERE =====================
- The title of this script.
- A brief explanation of this script, e.g. the purpose of this script, what 
can this script achieve or solve, what algorithms are used in this script...

- The name of author and the created date of this script.
===================== PLEASE WRITE HERE =====================
"""


# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Load data file and return two numpy arrays, including x: features and 
# y: labels
def load_data(path):
    print('Load data.')
    """
    - Read dataset description 'wine.names' in the first place to know how the 
    given data is arranged.
    - Use the function 'numpy.genfromtxt' to load the data.
    """
    data = np.genfromtxt(path, delimiter=',')
        
    # Print the number of samples
    n_samples = len(data)
    print('Number of samples:', n_samples)
    
    # Split the data into features and labels.
    # According to the dataset description, the first column is class 
    # identifier.
    x = data[:, 1:]
    y = data[:, 0].astype(int)
    
    return data, x, y 

# Get the number of samples in each class
def class_distribution(y):
    """
    - According to the dataset description, the given data consist of three 
    classes, namely 1, 2, and 3.
    - Please calculate the number of samples in each class. You may use the 
    function 'numpy.bincount'.
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    
    
    # ===================== PLEASE WRITE HERE =====================
       
    print('Number of samples in class_1:', n_class1)
    print('Number of samples in class_2:', n_class2)
    print('Number of samples in class_3:', n_class3)
    

# Split the data into training set and testing set
def split_dataset(x, y, testset_portion):
    print('Split dataset.')
    """
    - In order to prevent a ML model from seeing answers before making 
    predictions, data is usually divived into a training set and a testing 
    set. A training set will be used to train an ML model (e.g. a classifier or 
    a regressor) and a testing set will be used to evaluate the performance of 
    an ML model.
    - Please split the data (both x and y) into a training set and a testing 
    set according to the 'testset_portion'. That is, the testing set will 
    account for 'testset_portion' of the overall data. You may use the function
    'sklearn.model_selection.train_test_split'.    
    """
    # ===================== PLEASE WRITE HERE =====================
    
    
    
    # ===================== PLEASE WRITE HERE =====================
    
    return x_train, x_test, y_train, y_test
    

# Standardize the values of each feature dimension.
def feature_scaling(x_train, x_test):
    print('Feature scaling.')
    """
    - By observing the features 'x' in the 'Variable explorer', you will find
    that the values of some feature dimensions are 1, 2, or 1x, while some are 
    1xxx. It shows the values of each feature dimension are at different 
    levels. If these features are directly fed to an ML model, the result will 
    be dominate by the feature dimension with large values. As a result, the 
    process of feature scaling is necessary, which will standardize each 
    feature dimension with mean being 0 and stardard deviation being 1.
    - Please standardize both training set and testing set. You may use the 
    function 'sklearn.preprocessing.StandardScaler'.    
    """
    # ===================== PLEASE WRITE HERE =====================
    
    
    
    # ===================== PLEASE WRITE HERE =====================    

    return x_train_nor, x_test_nor

# Train a Naive Bayes classifier on x_train and y_train
def train(x_train, y_train):
    print('Start training.')
    """
    - After the preprocessing, we can now train an ML model. We will train a 
    Naive Bayes classifier, which is based on the Baysian Decision Theory.
    - Since the input features are continuous values, we choose Gaussian Naive 
    Bayes classifier.
    - Please use the function 'sklearn.naive_bayes.GaussianNB' to train a 
    classifier.    
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    
    
    # ===================== PLEASE WRITE HERE =====================    
        
    return clf

# Use the trained classifier to test on x_test
def test(clf, x_test):
    print('Start testing.')
    """
    - Now we can use the trained classifier to predict the classes on x_test
    - Likewise, please the function 'sklearn.naive_bayes.GaussianNB'.
    """
    # ===================== PLEASE WRITE HERE =====================
    
    
    
    # ===================== PLEASE WRITE HERE =====================
    
    return y_pred



# Main
if __name__=='__main__':
    # Some parameters
    path = 'wine.data'
    testset_portion = 0.2
    
    # Load data
    data, x, y = load_data(path)
    class_distribution(y)
    
    # Preprocessing
    x_train, x_test, y_train, y_test = split_dataset(x, y, testset_portion)
    x_train_nor, x_test_nor = feature_scaling(x_train, x_test)
    
    # Classification: train and test
    clf = train(x_train, y_train)
    y_pred = test(clf, x_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print('\nAccuracy:', round(acc, 3))