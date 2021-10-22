# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:51:02 2021

@author: Jannis
"""
import numpy as np
import math
import time
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.models import Model
from keras.models import Sequential

import gurobipy as gp
from gurobipy import GRB

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from Networks import BDNN
from keras.callbacks import ModelCheckpoint


def trainDNN(dnn, X_train, y_train, X_val, y_val, batchSize, epochs, regWeight):
    N=X_train.shape[1]
    LR = 0.001
    metric = ["accuracy"]
    
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    reg = tf.keras.regularizers.l2(regWeight)
    
    widths = dnn.getWidthsLayers()
    biasTerms = dnn.hasBiasTerms()
    
    model = Sequential()
    for l in range(1,len(widths)-1):
        model.add(Dense(units=widths[l], activation='relu', use_bias=biasTerms, kernel_regularizer=reg, bias_regularizer=reg))
        

    model.add(Dense(units=widths[len(widths)-1], activation='softmax', use_bias=biasTerms, kernel_regularizer=reg, bias_regularizer=reg))

    
    opt = Adam(lr=LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=metric)
    
    checkpoint_path = "DNN_Weights\\" + str(time.time()).replace(".","") + "cp.ckpt"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                  monitor='val_accuracy',
                              save_weights_only=True, 
                              save_best_only=True, 
                              verbose=1)
    
    history = model.fit(X_train,y_train, batch_size=batchSize, validation_data=(X_val,y_val), verbose=1, epochs=epochs, callbacks=[cp_callback])
    
    model.load_weights(checkpoint_path)
    model.summary()
    

    return history, model


def printStatus(stat):
    
    if stat  == GRB.OPTIMAL:
        pass
        #print('Model is optimal')
    elif stat  == GRB.INF_OR_UNBD:
        print('Model  is  infeasible  or  unbounded')
    elif stat  == GRB.INFEASIBLE:
        print('Model  is  infeasible')
    elif stat  == GRB.UNBOUNDED:
        print('Model  is  unbounded')
    else:
        print('Optimization  ended  with  status ' + str(stat))


def trainBDNN(X_train, y_train, X_val, y_val, bdnn, splittingMethod, maxSplits, regWeight, integerWeights, robustRadius, batchSize, randomBatch):
    list_sets = [[i for i in range(X_train.shape[0])]]
    
    numSplits = 0
    splitIndex = 0
    
    num_classes = y_train.shape[1]
    
    R=np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        R[i]=np.sum(np.absolute(X_train[i,:]))
        
    timeMIP = []
    trainLoss = []
    accuracy = []
    bestAccuracy = 0
    bestBDNN = BDNN(bdnn.getWidthsLayers(),bdnn.hasBiasTerms())
    
    optimal = False
    
    batchLIndex = 0
    batchUIndex = 0
    
    y_true_val = np.argmax(y_val, axis=1)
    
    
    while numSplits<maxSplits and not optimal:
        numSplits+=1
        
        splitSet = list_sets.pop(splitIndex)
        if splittingMethod=="k-means":
            set1, set2 = split_k_means(X_train,splitSet)
        elif splittingMethod == "svm":
            set1, set2 = split_svm(X_train,y_train,splitSet)
        
        list_sets.append(set1)
        list_sets.append(set2)
        
        weights = []
        currBDNN = BDNN(bdnn.getWidthsLayers(),bdnn.hasBiasTerms())
        start = time.time()
        if randomBatch:
            randIndex = np.random.randint(0,X_train.shape[0]-batchSize+1)
            batchIndices = np.arange(randIndex,randIndex+batchSize,1)
                
            obj = solveNetworkMIP(X_train, y_train, currBDNN, list_sets, R, regWeight, integerWeights, robustRadius, batchIndices)
        else:
            for j in range(0,X_train.shape[0],batchSize):
                batchIndices = np.arange(j,min(j+batchSize,X_train.shape[0]),1)
                
                obj = solveNetworkMIP(X_train, y_train, bdnn, list_sets, R, regWeight, integerWeights, robustRadius, batchIndices)
                
                y_pred = bdnn.predict(X_val)
                weightInAvg = accuracy_score(y_true_val,y_pred)
                weights.append(weightInAvg)
                for l in range(currBDNN.getNumLayers()):
                    currBDNN.setMatrix(l,currBDNN.getMatrix(l) + weightInAvg * bdnn.getMatrix(l))
                    currBDNN.setBiasVector(l,currBDNN.getBiasVector(l) + weightInAvg * bdnn.getBiasVector(l))
            
            for l in range(currBDNN.getNumLayers()):
                currBDNN.setMatrix(l,currBDNN.getMatrix(l)/sum(weights))
                currBDNN.setBiasVector(l,currBDNN.getBiasVector(l)/sum(weights))
            
        end = time.time()
        
        currLoss = np.zeros(X_train.shape[0])
        for i in range(X_train.shape[0]):
            u_pred = currBDNN.evaluate(X_train[i])
            for j in range(num_classes):
                #loss[i] = loss[i] + y_train[i,j] +  (1-2*y_train[i,j])*u[L-1][patternMap[i],j].x
                currLoss[i] = currLoss[i] + y_train[i,j] +  (1-2*y_train[i,j])*u_pred[j]
                
        totalLoss = np.sum(currLoss)
        
        timeMIP.append(end-start)
        trainLoss.append(totalLoss/X_train.shape[0])
        
        
        #Calculate Validation Accuracy
        y_pred = currBDNN.predict(X_val)
        currAccuracy = accuracy_score(y_true_val,y_pred)
        accuracy.append(currAccuracy)
        
        if currAccuracy > bestAccuracy:
            bestAccuracy = currAccuracy
            loss = currLoss[:]
            for l in range(bestBDNN.getNumLayers()):
                bestBDNN.setMatrix(l,currBDNN.getMatrix(l))
                bestBDNN.setBiasVector(l,currBDNN.getBiasVector(l))
        
            
        print("Split:", numSplits,"    Training Loss:",round(totalLoss/X_train.shape[0],2),"   Validation-Accuracy:", round(currAccuracy*100,3))
        
        
        maxLoss = -100
        for i in range(len(list_sets)):
            if len(list_sets[i])>1:
                lossSet = 0
                for j in list_sets[i]:
                    lossSet = lossSet + loss[j]
                if lossSet > maxLoss:
                    maxLoss = lossSet
                    splitIndex = i
                    
        if maxLoss == -100:
            optimal = True
                
    
    return timeMIP, trainLoss, accuracy, bestBDNN
                
        
        
def solveNetworkMIP(X_train, y_train, bdnn, list_sets, R, regWeight, integerWeights, robustRadius, batchIndices):
    
    ip = gp.Model("LinearizedIPModel")
    ip.setParam("OutputFlag",0)
    ip.setParam("TimeLimit", 7200)
    ip.setParam('MIPGap', 0.0001)
    
    eps = 0.00001
    
    
    widths = bdnn.getWidthsLayers()
    hasBias = bdnn.hasBiasTerms()
    
    m=X_train.shape[0]
    num_patterns=len(list_sets)
    num_classes = widths[len(widths)-1]
    L = len(widths)-1
    
    patternMap = -np.ones(m, dtype=int)
    for i in range(len(list_sets)):
        for j in list_sets[i]:
            patternMap[j] = i

    boundBias = 0
    if hasBias: boundBias = max(np.amax(R),np.amax(widths[1:len(widths)]))
    
    # Create variables
    u = []
    W = []
    b = []
    for l in range(1,len(widths)):
        u.append(ip.addVars(num_patterns,widths[l], vtype=GRB.BINARY, name = "u"+str(l)))
        if integerWeights:
            W.append(ip.addVars(widths[l],widths[l-1],lb=-1.0, ub=1.0, vtype=GRB.INTEGER, name = "W"+str(l)))
        else:
            W.append(ip.addVars(widths[l],widths[l-1],lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name = "W"+str(l)))
        if hasBias:
            if integerWeights:
                b.append(ip.addVars(widths[l], vtype=GRB.INTEGER, lb=-boundBias, ub=boundBias, name = "b"+str(l)))
            else:
                b.append(ip.addVars(widths[l], vtype=GRB.CONTINUOUS, lb=-boundBias, ub=boundBias, name = "b"+str(l)))
    xi =  ip.addVars(widths[1],widths[0],lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name = "xi")           
                
    #Set Start-Solution from already obtained bdnn weights
    xi_val = np.absolute(bdnn.getMatrix(0))
    xi.start=xi_val
    for l in range(1,len(widths)):
        W_start=bdnn.getMatrix(l-1)
        if hasBias: b_start = bdnn.getBiasVector(l-1)
        for i in range(W_start.shape[0]):
            if hasBias: b[l-1][i].start = b_start[i]
            for j in range(W_start.shape[1]):
                W[l-1][i,j].start = W_start[i,j]
               
    
    #Set objective
    lhs=""
    for i in batchIndices:
        for j in range(num_classes):
            lhs = lhs + (1-2*y_train[i,j])*u[L-1][patternMap[i],j]
            # if y_train[i,j]==1:
            #     lhs = lhs - 1 * u[L-1][patternMap[i],j]
    
    #Quadratic Regularizer
    if regWeight>0:
        for l in range(1,len(widths)):
            for i in range(widths[l]):
                if hasBias: lhs = lhs + regWeight * b[l-1][i] * b[l-1][i]
                for j in range(widths[l-1]):
                    lhs = lhs + regWeight * W[l-1][i,j] * W[l-1][i,j]
            
    ip.setObjective(lhs, GRB.MINIMIZE)
    
    #Constraints xi>=W1 , xi>=-W1 (to model xi=|W1|)
    for l in range(widths[1]):
        for j in range(widths[0]):
            lhs = ""
            lhs = 1 * xi[l,j] - 1 * W[0][l,j]
            ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0)
            
            lhs = 1 * xi[l,j] + 1 * W[0][l,j]
            ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0)
            
    
    #Constraints First Layer
    for i in batchIndices:
        for l in range(widths[1]):
            # Add constraints: W1x^i<=Mu^i + lam-eps
            lhs1=""
            lhs2=""
            for j in range(widths[0]):
                lhs1 = lhs1 + X_train[i,j] * W[0][l,j] + robustRadius * xi[l,j]
                lhs2 = lhs2 + X_train[i,j] * W[0][l,j] - robustRadius * xi[l,j]
                
            if hasBias:
                lhs1 = lhs1 + 1 * b[0][l]
                lhs2 = lhs2 + 1 * b[0][l]
                    
            lhs1 = lhs1 - (R[i]+boundBias+1+eps) * u[0][patternMap[i],l]
            lhs2 = lhs2 - (R[i]+boundBias+1+eps) * u[0][patternMap[i],l]
            ip.addConstr(lhs1, sense=GRB.LESS_EQUAL, rhs=-eps)
            
            # Add constraints: W1x^i>=M(u^i-1) + lam
            ip.addConstr(lhs2, sense=GRB.GREATER_EQUAL, rhs=-(R[i]+boundBias+1))
    
    #Constraint last layers      
    for k in range(1,L):
        for i in range(num_patterns):
            for l in range(widths[k+1]):
                # Add constraints: W1x^i<=Mu^i + lam-eps
                lhs=""
                for j in range(widths[k]):
                    lhs = lhs + u[k-1][i,j] * W[k][l,j]
                    
                if hasBias:
                    lhs = lhs + 1 * b[k][l]
                        
                lhs = lhs - (widths[k]+boundBias+1+eps) * u[k][i,l]
                ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-eps)
                
                # Add constraints: W1x^i>=M(u^i-1) + lam
                ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-(widths[k]+boundBias+1))
     
        
        
    ip.write("LinearizedIPModel.lp")
    # Optimize model
    ip.optimize()
    
    printStatus(ip.status)
    
    for l in range(1,len(widths)):
       W_opt=np.zeros((widths[l],widths[l-1])) 
       if hasBias : b_opt = np.zeros(widths[l]) 
       for i in range(W_opt.shape[0]):
           if hasBias : b_opt[i]=b[l-1][i].x 
           for j in range(W_opt.shape[1]):
               W_opt[i,j]=W[l-1][i,j].x
               
       bdnn.setMatrix(l-1,W_opt)
       if hasBias : bdnn.setBiasVector(l-1,b_opt)
           
    
    return ip.objVal + m

        
    
def split_k_means(X_train,splitSet):
    X = X_train[splitSet,:]
    
    kmeans = KMeans(n_clusters=2, n_init = 25).fit(X)
    
    set1 = []
    set2 = []
    
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i]==1:
            set1.append(splitSet[i])
        else:
            set2.append(splitSet[i])
            
    return set1, set2



def getAccuracy(X_test, y_test, bdnn):
    count = 0
    for i in range(X_test.shape[0]):
        y_pred = bdnn.evaluate(X_test[i,:])
        if np.argmax(y_test[i,:])==np.argmax(y_pred):
            count +=1
            
    return float(count) / X_test.shape[0]



            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    