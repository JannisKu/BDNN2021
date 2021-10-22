# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:07:00 2020

@author: Jannis
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


    

def predictBDNN(X_test,W1,w2,lam,v,eps):
    m=X_test.shape[0]
    y=np.zeros(m,dtype=int)

    for i in range(0,m,1):
        #Calculate predicted label for DataPoint i
        c=np.dot(W1,X_test[i])

        u=np.zeros(c.shape[0])
        for j in range(0,c.shape[0],1):
            if c[j]>=lam[0]:
                u[j]=v
        z=np.dot(w2,u)

        if z>=lam[1]:
            y[i]=1
        else:
            y[i]=0
            
    return y

def solveHeuristicBDNN(N,m,d2,X,y_cat,v,R,eps,timeLimit):
    # Create a new model
    ip1 = gp.Model("LayerProblem")
    ip2 = gp.Model("WeightProblem")
    
    ip1.setParam("TimeLimit", timeLimit)
    ip2.setParam("TimeLimit", timeLimit)
    
    y = np.argmax(y_cat, axis=1)
    
    maxSumU=0
    
    w2Last = 2*np.random.rand(d2)
    w2Last = w2Last-1
    uLast = np.ones((m,d2))
    lamLast = [0,0]
    

    # Create variables
    z1=ip1.addVars(m, vtype=GRB.BINARY, name="z")
    u=ip1.addVars(m,d2, vtype=GRB.BINARY, name="u")
    W1=ip1.addVars(d2,N,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="W1") 
    lam=ip1.addVars(2,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="lam")

    
    w2=ip2.addVars(d2,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w2")
    z2=ip2.addVars(m, vtype=GRB.BINARY, name="z")
    
    
    #Set objective
    lhs1=""
    lhs2=""
    countOnes = 0
    for i in range(0,m,1):
        if y[i]==0:
            lhs1 = lhs1 + 1 * z1[i]
            lhs2 = lhs2 + 1 * z2[i]
        elif y[i]==1:
            lhs1 = lhs1 - 1 * z1[i]
            lhs2 = lhs2 - 1 * z2[i]
            countOnes +=1
            
    ip1.setObjective(lhs1, GRB.MINIMIZE)
    ip2.setObjective(lhs2, GRB.MINIMIZE)
    
    
    for i in range(0,m,1):
        for l in range(0,d2,1):
            # Add constraints: W1x^i<=Mu^i + lam-eps
            lhs=""
            for j in range(0,N,1):
                lhs = lhs + X[i,j] * W1[l,j]
            
            lhs = lhs + -(R[i]+1+maxSumU+eps) * u[i,l]
            lhs = lhs + -1 * lam[0]
            ip1.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-eps, name="W1_L_" + str(i) + "_" + str(l))
            
            # Add constraints: W1x^i>=M(u^i-1) + lam
            lhs=""
            for j in range(0,N,1):
                lhs = lhs + X[i,j] * W1[l,j]
                
            lhs = lhs + -(R[i]+1+maxSumU) * u[i,l]
            lhs = lhs + -1 * lam[0]
            ip1.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-(R[i]+maxSumU+1), name="W1_G_" + str(i) + "_" + str(l))
          
    #IP1 Problem
    for i in range(0,m,1):
        # Add constraints: v w^2 u^i <=vd2z^i + lam -eps
        lhs=""
        for j in range(0,d2,1):
            lhs = lhs + (v*w2Last[j]) * u[i,j]
            
        lhs = lhs + -(v*d2+1+eps) * z1[i]
        lhs = lhs + -1 * lam[1]
        ip1.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-eps, name="w2_L_" + str(i))
        
        # Add constraints:  v w^2 u^i >= vd2(z^i-1) + lam
        lhs=""
        for j in range(0,d2,1):
            lhs = lhs + (v*w2Last[j]) * u[i,j]
            
        lhs = lhs + -(v*d2+1) * z1[i]
        lhs = lhs + -1 * lam[1]
        ip1.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-(v*d2+1), name="w2_G_" + str(i))
        
    #IP2 Problem
    for i in range(0,m,1):
        # Add constraints: v w^2 u^i <=vd2z^i + lam -eps
        lhs=""
        for j in range(0,d2,1):
            lhs = lhs + (v*uLast[i,j]) * w2[j]
            
        lhs = lhs + -(v*d2+1+eps) * z2[i]
        ip2.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=lamLast[1]-eps, name="w2_L_" + str(i))
        
        # Add constraints:  v w^2 u^i >= vd2(z^i-1) + lam
        lhs=""
        for j in range(0,d2,1):
            lhs = lhs + (v*uLast[i,j]) * w2[j]
            
        lhs = lhs + -(v*d2+1) * z2[i]
        ip2.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-(v*d2+1)+lamLast[1], name="w2_G_" + str(i))
    
    
    # Optimize model
    
    optimal = False
    
    uNew = np.zeros((m,d2))
    lamNew = [0,0]
    
    w2New = np.zeros(d2)
    W1_ret = np.zeros((d2,N))
    objVal = 100
    
    countObjValueEqual = 0
    
    ip1.write("Problem1Heuristic.lp")
    ip2.write("Problem2Heuristic.lp")
    
    numIterations=0
    
    while not optimal:
        numIterations+=1
        #Update coefficients w^2 in IP1
        for i in range(0,m,1):
            for j in range(0,d2,1):
                c0 = ip1.getConstrByName("w2_L_" + str(i))
                ip1.chgCoeff(c0, u[i,j], v*w2Last[j])
                
                c0 = ip1.getConstrByName("w2_G_" + str(i))
                ip1.chgCoeff(c0, u[i,j], v*w2Last[j])
               
            
        ip1.optimize()
        
        if ip1.status  == GRB.OPTIMAL:
            print('Model is optimal')
            for i in range(0,m,1):
                for j in range(0,d2,1):
                    uNew[i,j] = ip1.getVarByName(u[i,j].VarName).x
                    
            for i in range(0,d2,1):
                for j in range(0,N,1):
                    W1_ret[i,j] = ip1.getVarByName(W1[i,j].VarName).x
                    
                
            lamNew = [ip1.getVarByName(lam[0].VarName).x,ip1.getVarByName(lam[1].VarName).x]
            if objVal == ip1.objVal+countOnes:
                countObjValueEqual+=1
            else:
                countObjValueEqual=0
                objVal = ip1.objVal+countOnes
            print("u:")
            print(uNew)
            print("Lambda")
            print(lamNew)
            print("Objective Value: ",objVal)
        elif ip1.status  == GRB.INF_OR_UNBD:
            print('Model  is  infeasible  or  unbounded')
            break
        elif ip1.status  == GRB.INFEASIBLE:
            print('Model  is  infeasible')
            break
        elif ip1.status  == GRB.UNBOUNDED:
            print('Model  is  unbounded')
            break
        else:
            print('Optimization  ended  with  status ' + str(ip1.status))
            break
                
        if np.array_equal(uLast,uNew) and np.array_equal(lamLast,lamNew):
            break
        
        uLast = np.copy(uNew)
        lamLast = np.copy(lamNew)
        
        #Update Coefficients u^i and lambda in IP2
        for i in range(0,m,1):
            for j in range(0,d2,1):
                c0 = ip2.getConstrByName("w2_L_" + str(i))
                ip2.chgCoeff(c0, w2[j], v*uLast[i,j])
                ip2.setAttr("RHS", c0, lamLast[1]-eps)
                
                c0 = ip2.getConstrByName("w2_G_" + str(i))
                ip2.chgCoeff(c0, w2[j], v*uLast[i,j])
                ip2.setAttr("RHS", c0, -(v*d2+1)+lamLast[1]) 
                
        ip2.optimize()
        
        if ip2.status  == GRB.OPTIMAL:
            print('Model is optimal')

            for j in range(0,d2,1):
                w2New[j] = ip2.getVarByName(w2[j].VarName).x
            if objVal == ip2.objVal+countOnes:
                countObjValueEqual+=1
            else:
                countObjValueEqual=0
                objVal = ip2.objVal+countOnes
            print("w2:")
            print(w2New)
            print("Objective Value: ",objVal)
        elif ip2.status  == GRB.INF_OR_UNBD:
            print('Model  is  infeasible  or  unbounded')
            break
        elif ip2.status  == GRB.INFEASIBLE:
            print('Model  is  infeasible')
            break
        elif ip2.status  == GRB.UNBOUNDED:
            print('Model  is  unbounded')
            break
        else:
            print('Optimization  ended  with  status ' + str(ip2.status))
            break
        
        if np.array_equal(w2Last,w2New):
            break
        
        w2Last = np.copy(w2New)
        
        
        if countObjValueEqual == 100 or objVal == 0:
            optimal = True
    
    return W1_ret,w2Last,lamLast,objVal,numIterations

def solveExactBDNN(N,m,d2,X,y_cat,v,R,eps,timeLimit, U=0):
    # Create a new model
    ip = gp.Model("LinearizedIPModel")
    
    ip.setParam("TimeLimit", timeLimit)
    ip.setParam('MIPGap', 0.0001)
    
    y = np.argmax(y_cat, axis=1)
    
    if(not U==0):
        maxSumU=sum(U[1])
    else:
        maxSumU=0
    

    # Create variables
    z=ip.addVars(m, vtype=GRB.BINARY, name="z")
    u=ip.addVars(m,d2, vtype=GRB.BINARY, name="u")
    W1=ip.addVars(d2,N,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="W1") 
    w2=ip.addVars(d2,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w2")
    s=ip.addVars(m,d2,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="s")
    #lam=ip.addVars(2,lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="lam")
    
    lam=[0,0]
    
    
    #Set objective
    lhs=""
    countOnes = 0
    for i in range(0,m,1):
        if y[i]==0:
            lhs = lhs + 1 * z[i]
        elif y[i]==1:
            lhs = lhs - 1 * z[i]
            countOnes +=1
            
    ip.setObjective(lhs, GRB.MINIMIZE)
    
    for i in range(0,m,1):
        for l in range(0,d2,1):
            # Add constraints: W1x^i<=Mu^i + lam-eps
            lhs=""
            for j in range(0,N,1):
                if(U==0):
                    lhs = lhs + X[i,j] * W1[l,j]
                else:
                    lhs = lhs + X[i,j] * W1[l,j] + U[1][j] * mu[l,j]
                
                
            lhs = lhs + -(R[i]+1+maxSumU+eps) * u[i,l]
            lhs = lhs + -1 * lam[0]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-eps, name="W1_L_" + str(i) + "_" + str(l))
            
            # Add constraints: W1x^i>=M(u^i-1) + lam
            lhs=""
            for j in range(0,N,1):
                if(U==0):
                    lhs = lhs + X[i,j] * W1[l,j]
                else:
                    lhs = lhs + X[i,j] * W1[l,j] - U[1][j] * mu[l,j]
                
                
            lhs = lhs + -(R[i]+1+maxSumU+eps) * u[i,l]
            lhs = lhs + -1 * lam[0]
            ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-(R[i]+maxSumU+1), name="W1_G_" + str(i) + "_" + str(l))
            
    for i in range(0,m,1):
        # Add constraints: v sum s_j^i <=vd2z^i + lam -eps
        lhs=""
        for j in range(0,d2,1):
            lhs = lhs + v * s[i,j]
            
        lhs = lhs + -(v*d2+1+eps) * z[i]
        lhs = lhs + -1 * lam[1]
        ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-eps, name="w2_L_" + str(i))
        
        # Add constraints: v sum s_j^i >= vd2(z^i-1) + lam
        lhs=""
        for j in range(0,d2,1):
            lhs = lhs + v * s[i,j]
            
        lhs = lhs + -(v*d2+1) * z[i]
        lhs = lhs + -1 * lam[1]
        ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-(v*d2+1), name="w2_G_" + str(i))
    
    for i in range(0,m,1):
        for j in range(0,d2,1):
            # Add constraints:
            lhs= s[i,j] - u[i,j]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0.0, name="s1_" + str(i) + "_" + str(j))
            
            # Add constraints:
            lhs= s[i,j] +  u[i,j]
            ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0.0, name="s2_" + str(i) + "_" + str(j))
            
            # Add constraints:
            lhs= s[i,j] - w2[j] + u[i,j]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=1.0, name="s3_" + str(i) + "_" + str(j))
            
            # Add constraints:
            lhs= s[i,j] - w2[j] - u[i,j]
            ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-1.0, name="s4_" + str(i) + "_" + str(j))
    
    # for i in range(0,m,1):
    #     # Add constraints:
    #     lhs= xi[i] - z[i]
    #     ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=-y[i], name="xi1_" + str(i))
        
    #     # Add constraints:
    #     lhs= xi[i] + z[i]
    #     ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=y[i], name="xi2_" + str(i))
        
    if(not U==0):
        for l in range(0,d2,1):
            for j in range(0,N,1):
                # Add constraints:
                lhs = mu[l,j] - W1[l,j]
                ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0.0, name="mu1_" + str(l) + "_" + str(j))
                
                # Add constraints:
                lhs= mu[l,j] + W1[l,j]
                ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0.0, name="mu2_" + str(l) + "_" + str(j))
    
    
    #ip.write("LinearizedIPModel.lp")
    # Optimize model
    ip.optimize()
    
    if ip.status  == GRB.OPTIMAL:
        print('Model is optimal')
    elif ip.status  == GRB.INF_OR_UNBD:
        print('Model  is  infeasible  or  unbounded')
    elif ip.status  == GRB.INFEASIBLE:
        print('Model  is  infeasible')
    elif ip.status  == GRB.UNBOUNDED:
        print('Model  is  unbounded')
    else:
        print('Optimization  ended  with  status ' + str(ip.status))
    
    W1_ret = np.zeros((d2,N))
    w2_ret = np.zeros(d2)
    for i in range(0,d2,1):
        w2_ret[i] = ip.getVarByName(w2[i].VarName).x
        for j in range(0,N,1):
            W1_ret[i,j] = ip.getVarByName(W1[i,j].VarName).x
            
    #lam_ret = [ip.getVarByName("lam[0]").x,ip.getVarByName("lam[1]").x]
    lam_ret = [0,0]
    objVal = ip.objVal + countOnes
            
    
    return W1_ret,w2_ret,lam_ret,objVal

