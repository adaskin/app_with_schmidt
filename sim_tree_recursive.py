#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this simulates tree to find number of non-zero path. 
It DOES NOT STORE the matrices
Created on Wed Dec 14 18:28:31 2022

@author: adaskin
"""
import  numpy as np
import  matplotlib.pyplot as plt


def find_childs(a,sprev, n0path, listofprobs):
    if a.size <= 2: 
        return a;
    n0 = 0
    n1 = 0
    A = a.reshape(2, int(a.size/2))
    #print("shape of A:", A.shape, A.size)
    u, s, v = np.linalg.svd(A,full_matrices=False)
    minprob = 0.025

    #print("size:",int(np.log2(v[0,:].size)), s[0]**2)
    if (v[0,:].size == 2):
        if (sprev*s[0] >  minprob):
            n0path += 1
        
        if (sprev*s[1] >  minprob):
            n0path += 1
        #print("\n\nzero_path:", n0path)
        #print("the matrix u\n", np.round(u,2))
        #print("the matrix s\n", np.round(s,2))
        #print("square s0+s1\n", s[0]**2+s[1]**2)
        
       # print("{}, {},".\
        #    format(sprev*s[0], sprev*s[1]))
        listofprobs.append(sprev*s[0])
        listofprobs.append(sprev*s[1])
    elif v[0,:].size > 2:
        #print("=======find child v0:sprev*s[0]:{}", sprev*s[0])
        #if(sprev*s[0]>0.000001):
        n0 = find_childs(v[0,:],sprev*s[0], n0path, listofprobs)

        #print("=======find child v1==========sprev*s[1]:", sprev*s[1])
        #if(sprev*s[1]>0.000001):
        n1 = find_childs(v[1,:],sprev*s[1], n0path, listofprobs)
        
    n0path += n0+n1

    return n0path

def qft(N):
    w = np.exp(2*np.pi*1j/N)
    A = np.zeros([N,N], dtype='complex')
    for i in range(0, N):
        for j in range(0, N):
            A[i,j] = w**(j*i)/N
    print("shape of A:", A.shape)
    return A


n = 20
N = 2**n
psi = 0
scase = 2
sprev = 1
n0path = 0
Nsquare = N**2
if scase == 1:#qft
    A = qft(N)
    a = A.flatten()
    psi = a
elif scase == 2:#random
    b = np.random.randn(1,N)
    b = b/np.linalg.norm(b)
    psi = b
else:#grover
    c =  np.eye(N,N).flatten()/np.sqrt(N)
    c[-1] = -c[-1]
    psi = c

listofprobs = []
print('=================================================================\n\n')
n0path = find_childs(psi,sprev,0, listofprobs)
print("n0path:{}".format(n0path))

fig, ax = plt.subplots()
values, bins, bars = ax.hist(listofprobs)

ax.set_xlabel("probability (coefficient) of the path")
ax.set_ylabel("Number of paths")
ax.set_title('n:{}-qubits '.format(n))
ax.bar_label(bars, fontsize=9, color='red')



