#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:47:57 2022

@author: adaskin
"""
import numpy as np
from numpy import linalg as la
epsilon = 0.0000001
I = np.array([[1,0],[0,1]], float);
H = np.array([[1, 1],[1, -1]],float)#*1/np.sqrt(2)*;
Z = np.array([[1,0],[0,-1]], float);
X = np.array([[0,1],[1,0]], float);

N = 16;
#[L, A] = la.eig(np.random.rand(N,N))
#a =A.flatten()

a = np.random.rand(N,1)
a = a/la.norm(a)
A2 = a.reshape([2,int(N/2)])

#V1@np.diag(s)@V2h = (V1*s)@V2h
[U, s, Vh] = la.svd(A2, full_matrices=False);  #Vh = Vh[0:2,:]
assert(la.norm(A2- (U*s)@Vh, ord=1) < epsilon)

V = Vh.transpose() #WARNING V and Vh refers to same memory
r = s[0]*np.kron(U[:,0],V[:,0])+s[1]*np.kron(U[:,1],V[:,1])
a.flatten() - r