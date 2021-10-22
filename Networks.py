# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:18:15 2021

@author: Jannis
"""
import numpy as np


class BDNN:
    __numNetworks = 0
    
    def __init__(self, layer_widths=[0], bias_terms=True):
        BDNN.__numNetworks +=1
        
        self.__id = BDNN.__numNetworks
        self.__n = layer_widths[0]
        self.__L = len(layer_widths)-1
        self.__widths = layer_widths[:]
        self.__bias = bias_terms
        
        
        self.__matrices = []
        self.__biasVectors = []
        for l in range(self.__L):
            self.__matrices.append(np.zeros((self.__widths[l+1],self.__widths[l]),dtype=float))
            if self.__bias:
                self.__biasVectors.append(np.zeros(self.__widths[l+1],dtype=float))
    
    def getNumLayers(self):
        return self.__L
    
    def getWidthsLayers(self):
        return self.__widths[:]
    
    def hasBiasTerms(self):
        return self.__bias
                
    def printMatrices(self):
        for l in range(self.__L):
            print(self.__matrices[l])
            
    def printBiasVectors(self):
        for l in range(self.__L):
            print(self.__biasVectors[l])
            
    
    def evaluate(self,x):
        c = x[:]            
        for l in range(self.__L):
            c = np.dot(self.__matrices[l],c)
            if self.__bias:
                c = c + self.__biasVectors[l]
            c[c>0]=1
            c = np.maximum(c,0)
            
        return c  
    
    def predict(self,X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i]=np.argmax(self.evaluate(X[i,:]))
            
        return y_pred
         
    def setBiasVector(self,l,b):
        if b.shape[0]!=self.__biasVectors[l].shape[0]:
            print("Error in DNN.setMatrix(): Shapes of Matrices not the same!")
        else:
            np.copyto(self.__biasVectors[l],b)
        
    def setMatrix(self,l, W):
        if W.shape[0]!=self.__matrices[l].shape[0] or W.shape[1]!=self.__matrices[l].shape[1]:
            print("Error in DNN.setMatrix(): Shapes of Matrices not the same!")
        else:
            np.copyto(self.__matrices[l],W)
            
    def getMatrix(self,l):
        if l<len(self.__matrices):
            return self.__matrices[l]
        else:
            return -1
        
    def getBiasVector(self,l):
        if l<len(self.__biasVectors):
            return self.__biasVectors[l]
        else:
            return -1
            
    @staticmethod
    def getNumNetworks():
        return BDNN.__numNetworks
                
            