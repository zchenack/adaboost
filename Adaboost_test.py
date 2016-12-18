# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:15:23 2016

@author: dell

used for Adaboost
"""

from __future__ import division
import numpy as np
import csv

def WeakClf(X,data_cat):
    '''
    weak classifier
    X is the only axis data
    data_cat is the data catogory
    return thred is the threshold to divide data
    return flag, flag=1, direction of inequality is greater than else smaller than
    '''
    ind_1 = np.where(data_cat==1)
    ind_0 = np.where(data_cat==0)
    max_1 = max(X[ind_1])
    min_1 = min(X[ind_1])
    max_0 = max(X[ind_0])
    min_0 = min(X[ind_0])
    flag = 1
    if max_1 < max_0:
        flag = 0
    thred = (min([max_1,max_0])+max([min_1,min_0]))/2
    return thred,flag

def Pred(data_feature,n,thred,flag):
    '''
    use the classififer to predict
    n is the n-th aixs
    '''
    N = data_feature.shape[0]
    dat = data_feature[:,n]
    pre_out = []
    for i in range(N):
        if flag==1:
            if dat[i] >= thred:
                pre_out.append(1)
            else:
                pre_out.append(0)
        else:
            if dat[i] >= thred:
                pre_out.append(0)
            else:
                pre_out.append(1)
    return np.array(pre_out)

def Pred_M(data_feature,n_dim,thred_dim,flag_dim,alpha):
    '''
    use the total Model to predict
    '''
    L = len(n_dim)
    N = data_feature.shape[0]
    if L==1:
        Pre_out = Pred(data_feature,n_dim,thred_dim,flag_dim)
        return Pre_out
    else:
        Pre_out = np.zeros(N)
        for i in range(L):
            Pre_out += alpha[i]*Pred(data_feature,n_dim[i],thred_dim[i],flag_dim[i])
    for i in range(N):
        if Pre_out[i]>0.5:
            Pre_out[i] = 1
        else:
            Pre_out[i] = 0
    return Pre_out

def Cal_e(pre_out,data_cat,W):
    '''
    used for calculate the predict error
    '''
    N = data_cat.shape[0]
    e = 0
    for i in range(N):
        e += W[i]*abs(pre_out[i]-data_cat[i])
    return e

def Cal_alpha(e):
    '''
    calculate the error
    '''
    return float(0.5*np.log((1.0-e)/max(e,1e-16)))

def Update_W(W,alpha,data_cat,pre_out):
    '''
    update the weight
    '''
    Z = Cal_Z(W,alpha,data_cat,pre_out)
    N = data_cat.shape[0]
    WW = []
    for i in range(N):
        if data_cat[i]==pre_out[i]:
            WW.append(W[i]/Z*np.exp(-alpha))
        else:
            WW.append(W[i]/Z*np.exp(alpha))
    return np.array(WW)

def Cal_Z(W,alpha,data_cat,pre_out):
    '''
    calculate the Z
    '''
    N = data_cat.shape[0]
    Z = 0
    for i in range(N):
        if data_cat[i]==pre_out[i]:
            Z += W[i]*np.exp(-alpha)
        else:
            Z += W[i]*np.exp(alpha)
    return Z
    
#load the data
csvfile = file('pimadiabetes.csv','rb')
reader = csv.reader(csvfile)
#feature data
data_feature = []
#catogory data
data_cat = []
cnt = 0
for line in reader:
    cnt += 1
    if cnt==1:
        continue
    data_feature.append(map(float,line[:-1]))
    data_cat.append(int(line[-1]))

data_feature = np.array(data_feature)
data_cat = np.array(data_cat)

#The number of aixs
n_axis = data_feature.shape[1]
m = 0
M = 3
N = cnt -1
W = 1/N*np.ones(N)

Alpha = []
Axis_Ind = []
n_dim = []
thred_dim = []
flag_dim = []
while m<M:
    m +=1
    E = np.ones(n_axis)
    Thred = np.ones(n_axis)
    Flag = np.ones(n_axis)
    for j in range(n_axis):
        if j in Axis_Ind:
            continue
        thred,flag = WeakClf(data_feature[:,j],data_cat)
        pre_out = Pred(data_feature,j,thred,flag)
        e = Cal_e(pre_out,data_cat,W)
        E[j] = e
        Thred[j] = thred
        Flag[j] = flag
    ee = min(E)
    ind = list(E).index(ee)
    n = ind
    Axis_Ind.append(n)
    tthred = Thred[n]
    fflag = Flag[n]
    print "%d round\n"%m 
    print "feature thresholded:%d\n"%n
    print "threshold is %f\n"%tthred
    if fflag==1:
        print "direction of inequality is greater than\n"
    else:
        print "direction of inequality is smaller than\n"
    ppre_out = Pred(data_feature,n,tthred,fflag)
    alpha = Cal_alpha(ee)
    Alpha.append(alpha)
    n_dim.append(n)
    thred_dim.append(tthred)
    flag_dim.append(fflag)
    Pre_out = Pred_M(data_feature,n_dim,thred_dim,flag_dim,Alpha)
    accuracy = (N -sum(abs(Pre_out-data_cat)))/N
    print "Accuracy is %.8f\n"%accuracy 
    W = Update_W(W,alpha,data_cat,ppre_out)