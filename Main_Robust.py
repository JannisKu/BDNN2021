# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:44:32 2021

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
import Functions as f
import keras
import sklearn.preprocessing
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical



fracTest=0.5
fracValidation = 0.5
epochs = 20
batchSizeDNN = 64
batchSizeBDNN = 64
randomBatch = True
splittingMethod = "k-means"  # k-means or k-median
biasTerms = True
integerWeights = True
maxSplits = 5
regWeight = 0 # Weight for Regularizer
epsDefend = 0.1
numLayers = 2 #number of hidden layers
widthLayers = 50
dataset = "breast_cancer"


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
    #X = sklearn.preprocessing.normalize(X, norm='l1', axis=0)
    print("Percentage Class 1:",np.sum(y)/y.shape[0])
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

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Create Neural Network
layers = [n]
for i in range(numLayers):
    layers.append(widthLayers)
layers.append(y.shape[1])

DNN1 = BDNN(layer_widths=layers, bias_terms= biasTerms)

fileName = dataset + "_robust_layer" + str(numLayers) + "_width" + str(widthLayers)
fileNameScores = fileName + "_scores.csv"
fileNameHistory = fileName + "_history.csv"

hyperParameters = [fileName,fracTest,fracValidation,epochs,maxSplits,batchSizeDNN,batchSizeBDNN,regWeight,epsDefend]
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
    
    start = time.time()
    timeMIP, trainLoss, accuracy, bestBDNN = f.trainBDNN(X_train, y_train, X_val, y_val, DNN1, splittingMethod, maxSplits, regWeight, integerWeights, epsDefend, batchSizeBDNN, randomBatch)
    end = time.time()
    
    runtimeBDNN = end-start
    
    print("\n########################Test-Set Evaluation #################")
    for epsAttack in [0,0.5,1.0,1.5,2.0]:
        print("Attack:",epsAttack)
        attackVector = -np.ones(X_test.shape) + np.random.rand(X_test.shape[0], X_test.shape[1])*2
        attackVector = epsAttack * np.sign(attackVector)

        X_test_attacked = X_test + attackVector
        ###################### DNN  #######################################
        y_pred = np.argmax(model.predict(X_test_attacked),axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\nDNN:")
        accuracyDNN = accuracy_score(y_true,y_pred)
        print("Test Accuracy DNN:",round(accuracyDNN*100,3))
        print("Confusion Matrix:\n",confusion_matrix(y_true,y_pred))
        
        df = pd.DataFrame([epsDefend,epsAttack,accuracyDNN])
        df = df.transpose()
        df.to_csv("DNN_" + fileNameScores, sep=';', mode='a', header=False, index=False)
        
        ######################  BDNN  ################################
        y_pred = bestBDNN.predict(X_test_attacked)
        
        print("\nBDNN:")
        accuracyBDNN = accuracy_score(y_true,y_pred)
        print("Test Accuracy BDNN:",round(accuracyBDNN*100,3))
        print("Confusion Matrix:\n",confusion_matrix(y_true,y_pred))
        
        df = pd.DataFrame([epsDefend,epsAttack,accuracyBDNN])
        df = df.transpose()
        df.to_csv("BDNN_" + fileNameScores, sep=';', mode='a', header=False, index=False)
        
