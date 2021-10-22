# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:31:14 2020

@author: Jannis
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random as r
import csv
import time
import os
from Networks import BDNN
from Networks import ConvBDNN
import Functions as f
import sklearn
import tensorflow
import sys

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical



fracTest=0.5
fracValidation = 0.5
epochs = 30
batchSizeDNN = 64
batchSizeBDNN = 64
randomBatch = True
splittingMethod = "k-means"  # k-means or k-median
biasTerms = True
integerWeights = True
maxSplits = 20
regWeight = 0 # Weight for Regularizer
robustRadius = 0.0
numLayers = 1  #number of hidden layers
widthLayers = 100
dataset = "digits"


if dataset=="breast_cancer":
    targetColumn = 9
    data = np.genfromtxt('Data\\breast-cancer-wisconsin.csv',dtype = float, delimiter=';')
    y=data[:,targetColumn]
    X=data[:,np.delete(np.arange(0,data.shape[1],1),targetColumn)]
elif dataset=="credit_card":
    targetColumn = 23
    data = np.genfromtxt('Data\\default_of_credit_card_clients.csv',dtype = float, delimiter=';')
    y=data[:,targetColumn]
    X=data[:,np.delete(np.arange(0,data.shape[1],1),targetColumn)]
elif dataset=="MNIST":
    (X, y), (X_test, y_test) = keras.datasets.mnist.load_data()
    X = X.reshape(X.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
    y_test = to_categorical(y_test)
elif dataset=="california_housing":
    X,y = fetch_california_housing(return_X_y=True)
    median_y = np.median(y)
    y[y<=median_y]=0
    y[y>median_y]=1
    X = np.array(X)
    y=np.array(y)
elif dataset == "digits":
    X,y = load_digits(return_X_y=True)
    X = np.array(X)
    y=np.array(y)
elif dataset == "iris":
    X,y = load_iris(return_X_y=True)
    X = np.array(X)
    y=np.array(y)
elif dataset=="boston_housing":
    X,y = load_boston(return_X_y=True)
    median_y = np.median(y)
    y[y<=median_y]=0
    y[y>median_y]=1
    X = np.array(X)
    y=np.array(y)
    
y = to_categorical(y)
n=X.shape[1]

print("Number of Instances:", X.shape[0])
print("Number of Attributes:", X.shape[1])
print("Number of Classes:",y.shape[1])
for i in range(y.shape[1]):
    print("Fraction of Class ", i+1,":",np.count_nonzero(y,axis=0)[i]/y.shape[0])
    
sys.exit()

#Create Neural Network
layers = [n]
for i in range(numLayers):
    layers.append(widthLayers)
layers.append(y.shape[1])

DNN1 = BDNN(layer_widths=layers, bias_terms= biasTerms)

fileName = dataset + "_bias" + str(int(biasTerms)) + "_intWeights" + str(int(integerWeights)) + "_randBatch" + str(int(randomBatch)) \
    + "_layer" + str(numLayers) + "_width" + str(widthLayers)
fileNameScores = fileName + "_scores.csv"
fileNameHistory = fileName + "_history.csv"

hyperParameters = [fileName,fracTest,fracValidation,epochs,maxSplits,batchSizeDNN,batchSizeBDNN,regWeight,robustRadius]
df = pd.DataFrame(hyperParameters)
df = df.transpose()
df.to_csv("Hyperparameters.csv", sep=';', mode='a', header=False, index=False)

for i in range(10):

    if dataset=="MNIST":
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=fracValidation, shuffle=True)
        
    else:
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=fracTest, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=fracValidation, shuffle=True)
    
    print("X_train.shape:",X_train.shape)
    print("y_train.shape:",y_train.shape)
    print("X_val.shape",X_val.shape)
    print("y_val.shape",y_val.shape)
    print("X_test.shape",X_test.shape)
    print("y_test.shape",y_test.shape)
    
    start = time.time()
    history, model = f.trainDNN(DNN1, X_train, y_train, X_val, y_val, batchSizeDNN, epochs, regWeight)
    end = time.time()
    runtimeDNN = end-start
    # print('Confusion Matrix :')
    # print(confusion_matrix(y_true,y_pred))
    # print(classification_report(y_true,y_pred))
    
    start = time.time()
    timeMIP, trainLoss, accuracy, bestBDNN = f.trainBDNN(X_train, y_train, X_val, y_val, DNN1, splittingMethod, maxSplits, regWeight, integerWeights, robustRadius, batchSizeBDNN, randomBatch)
    end = time.time()
    
    runtimeBDNN = end-start
    
    print("\n########################Test-Set Evaluation #################")
    ###################### DNN  #######################################
    y_pred = np.argmax(model.predict(X_test),axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    #Classification Report
    df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
    df.to_csv("DNN_" + fileNameScores, sep=';', mode='a')
    #Time
    df = pd.DataFrame(["time_total",runtimeDNN])
    df = df.transpose()
    df.to_csv("DNN_" + fileNameScores, sep=';', mode='a', header=False, index=False)
    
    #History
    history.history['val_accuracy'].insert(0,"val_accuracy")
    df = pd.DataFrame(history.history['val_accuracy'])
    df = df.transpose()
    df.to_csv("DNN_" + fileNameHistory, sep=';', mode='a', header=False, index=False)
    history.history['loss'].insert(0,"train_loss")
    df = pd.DataFrame(history.history['loss'])
    df = df.transpose()
    df.to_csv("DNN_" + fileNameHistory, sep=';', mode='a', header=False, index=False)
    
    print("\nDNN:")
    print("Test Accuracy DNN:",round(accuracy_score(y_true,y_pred)*100,3))
    print("Confusion Matrix:\n",confusion_matrix(y_true,y_pred))
    
    ######################  BDNN  ################################
    y_pred = bestBDNN.predict(X_test)
    
    print("\nBDNN:")
    print("Test Accuracy BDNN:",round(accuracy_score(y_true,y_pred)*100,3))
    print("Confusion Matrix:\n",confusion_matrix(y_true,y_pred))
    
    #Classification Report
    df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
    df.to_csv("BDNN_" + fileNameScores, sep=';', mode='a')
    #Time
    df = pd.DataFrame(["time_total",runtimeBDNN])
    df = df.transpose()
    df.to_csv("BDNN_" + fileNameScores, sep=';', mode='a', header=False, index=False)
    
    
    #History
    accuracy.insert(0,"val_accuracy")
    df = pd.DataFrame(accuracy)
    df = df.transpose()
    df.to_csv("BDNN_" + fileNameHistory, sep=';', mode='a', header=False, index=False)
    trainLoss.insert(0,"train_loss")
    df = pd.DataFrame(trainLoss)
    df = df.transpose()
    df.to_csv("BDNN_" + fileNameHistory, sep=';', mode='a', header=False, index=False)
    timeMIP.insert(0,"time")
    df = pd.DataFrame(timeMIP)
    df = df.transpose()
    df.to_csv("BDNN_" + fileNameHistory, sep=';', mode='a', header=False, index=False)
    
    
    


