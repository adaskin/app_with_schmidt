#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:00:15 2023

@author: adaskin
"""

import numpy as np
import matplotlib.pyplot as plt
from sim_tree import generate_tree_elements,sum_of_nonzeropaths
import sklearn.datasets as datasets
import scipy.sparse
import scipy.linalg  

def rGate(angle):
    ''' rotation-y gate
    '''
    R = np.zeros([2,2])
    angle = angle/2
    R[0,0] = np.cos(angle);R[0,1] = np.sin(angle); 
    R[1,0] = -np.sin(angle); R[1,1] = np.cos(angle);
    return R


def cRgate(angle):
    R = np.zeros([2,2])
    
    angle = angle/2
    R[0,0] = np.cos(angle);R[0,1] = np.sin(angle); 
    R[1,0] = -np.sin(angle); R[1,1] = np.cos(angle);
    CR =  np.eye(4,4)
    CR[2:4,2:4] = R
    return CR
def unitaryAtOddLevel(n, theta):
    '''
    generates a kronecker of n/2 CR gates defined 
    by the parameter vector theta

    '''
    
    U =  np.eye(2,2)# no gate on the 1st qubit
    ngates = int(np.ceil(n/2))
    for i in range(0,ngates-1):
        #CR on (i--i+1)
        CR = cRgate(theta[i])
        U = np.kron(U,CR)
    #the last qubit
    U = np.kron(U, np.eye(2,2) )
    return U


def unitaryAtEvenLevel(n, theta):

    '''
    generates a kronecker of n/2 CR gates defined 
    by the parameter vector theta

    '''
    
    U = np.eye(1,1)
    ngates = int(np.ceil(n/2))
    for i in range(0,ngates):
        #CR on (i--i+1)
        CR = cRgate(theta[i])
        U = np.kron(U,CR)
    return U

def singleGatesOnAll(n,theta):
    ''' 
    assigns single gates 
    on each qubit with given angle
    '''
    U = np.eye(1,1)
    ngates = n
    for i in range(0,ngates):
        #R on (i--i+1)
        R = rGate(theta[i])
        U = np.kron(U,R)
    return U

def random_vqc(n,level):
    '''
    generates random VQC
    return
        U: unitary matrix for VQC
    parameters
        n: #qubits
        level: number of levels in vqc circuit
    '''
    U = np.eye(2**n,2**n)
    rng = np.random.default_rng()
    n2 = int(np.ceil(n/2))
    for lvl in range(level):
        print('level:',lvl, level)
        theta = rng.normal(size=(n))    
        Usingles = singleGatesOnAll(n,theta)
        print('Singles=====================')
        U = U@Usingles

        theta = rng.normal(size=(n2))
        if lvl%2 == 0:
            Ulevel = unitaryAtEvenLevel(n,theta)
            print('Evenlevel=====================')
        else:
            Ulevel = unitaryAtOddLevel(n,theta)
            print('Oddlevel=====================')
        
        U = U@Ulevel
        
    return U



if __name__ == "__main__" :
    n = 8 #number of qubits should be even
    level = 16
    filename = 'vec(VQC)'+ ', circuit_depth: {}'.format(level)
    N = 2**n
    U = random_vqc(n, level)  


    psi = U.flatten()
        
    psi_norm = np.linalg.norm(psi)
    psi = psi/psi_norm
    (usv_tree, nonzeropath) = generate_tree_elements(psi)
    L = []
    for i in range(int(psi.size/2)-1, psi.size-1):
        if isinstance(usv_tree[i], tuple):
            u, s, v, sprev = usv_tree[i]
            print("shape:", v.shape)
            L.append(usv_tree[i][3])
        else:
            L.append(0)
            
    threshold = max(L)/2
    fig, ax = plt.subplots()
    ax.axvline(threshold, linestyle='--')
    values, bins, bars = ax.hist(L)

    ax.set_xlabel("probability (coefficient) of the path")
    ax.set_ylabel("Number of paths")
    ax.set_title('n: {}-qubits, data: {} '.format(int(np.log2(psi.size)), filename))
    ax.bar_label(bars, fontsize=9, color='red')


    p, npaths = sum_of_nonzeropaths(usv_tree,threshold)
    print(np.linalg.norm(p-psi))
    print(np.linalg.norm(p-psi,1))
    print(np.dot(psi,p))
    #X_tensor = psi_norm*p.reshape(N,N)
    norm_of_diff = np.linalg.norm(psi-p)
    print("norm of diff:",norm_of_diff)
    ax.text(0.75, 0.75, 'norm of diff={:5.2E}'.format(norm_of_diff), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)
    #print("mean error matrix unnormalized:",np.mean((X_tensor-G)**2))
    plt.savefig('vqc{}qubits{}depth.eps'.format(n, level), bbox_inches='tight')
    plt.savefig('vqc{}qubits{}depth.png'.format(n, level),bbox_inches='tight')