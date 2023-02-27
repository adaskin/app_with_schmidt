#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:28:31 2022
this simulates tree to find number of non-zero path. 
It DOES  STORE the matrices u,s,v in each node: 
that is why it is slow! 
@author: adaskin
"""
import  numpy as np
import  matplotlib.pyplot as plt


def qft(N):
    '''
    generates quantum fourier transform matrix of dimension N=2^n
    N: dimension
    '''
    w = np.exp(2*np.pi*1j/N)
    A = np.zeros([N,N], dtype='complex')
    for i in range(0, N):
        for j in range(0, N):
            A[i,j] = w**(j*i)/N
    print("shape of A:", A.shape)
    return A

def find_childs(avec):
    if avec.size <= 2: 
        return avec;
    A = avec.reshape(2, int(avec.size/2))
    #print("shape of A:", A.shape, A.size)
    u, s, v = np.linalg.svd(A,full_matrices=False)
      

    return u,s,v
###########################################################
###########################################################
def recursive_find_childs(avec,sprev, n0path, listofprobs):
    '''
    recursive function to find Schmidt decomposition
    '''
    if avec.size <= 2: 
        return avec;
    n0 = 0
    n1 = 0
    u, s, v = find_childs(avec)
    minprob = 0.025

    #print("size:",int(np.log2(v[0,:].size)), s[0]**2)
    if (v[0,:].size == 2):
        if (sprev*s[0] >  minprob):
            n0path += 1
        
        if (sprev*s[1] >  minprob):
            n0path += 1
        listofprobs.append(sprev*s[0])
        listofprobs.append(sprev*s[1])
    elif v[0,:].size > 2:
        #print("=======find child v0:sprev*s[0]:{}", sprev*s[0])
        #if(sprev*s[0]>0.000001):
        n0 = recursive_find_childs(v[0,:],sprev*s[0], n0path, listofprobs)

        #print("=======find child v1==========sprev*s[1]:", sprev*s[1])
        #if(sprev*s[1]>0.000001):
        n1 = recursive_find_childs(v[1,:],sprev*s[1], n0path, listofprobs)
        
    n0path += n0+n1

    return n0path
###########################################################

###########################################################
###########################################################
def generate_tree_elements(avec):
    ''' 
    generates Schmidt decomp tree
    returns a list of u, s,v, sprev and number of nonzeropath
    sprev: the prob along the path
    return usv_tree:
    a list of nodes
    the nodes in usv_tree: |0|1|2|3|4|...|14|empty]
               0
              /\
             1 2
            /\ /\
           3 4 5 6   
    '''
    N =  avec.size
    if N <= 2: 
        return avec;  
    nonzeropath = 0
    #list of (u,s,v)
    usv_tree = [0]*(N-1)
    usv_tree[0] = (1,1,avec,1)
    level = 0 
    u,s,v = find_childs(avec)

    #left child, sprev is the combination of schmidt coeff along the path
    sprev = 1;
    usv_tree[1] = (u[:,0],s[0],v[0], sprev*s[0])
    #right child, sprev is the combination of schmidt coeff along the path
    usv_tree[2] = (u[:,1],s[1],v[1], sprev*s[1])

    while(v.shape[1] > 2): #level child id
        level += 1
        istart = 2**(level)-1
        iend = istart+2**(level)
        inode = iend
        for i in range(istart, iend):
            # sprev is the combination of schmidt coefficients along the path
            sprev = 0
            if type(usv_tree[i]) != int:
                sprev = usv_tree[i][3]

            #if sprev > 0.0000001: #if prob is too small, skip
            u,s, v = find_childs(usv_tree[i][2])
            #left child, 
            usv_tree[inode] = (u[:,0], s[0], v[0], sprev*s[0])
            #rightchild
            usv_tree[inode+1] = (u[:,1], s[1], v[1],sprev*s[1])
            if v.shape[1] == 2:
                nonzeropath += 1
            print("svd node:{}, addnodes:{},{}"
                .format(i, inode, inode+1))

            inode += 2 
            
    return usv_tree, nonzeropath
###########################################################
###########################################################



#############################################################

def check_usv_tree(usv_tree):
    ''' check the correctness of the tree nodes
    '''
    for i in range(0, int(len(usv_tree)/2)):
        ui,si, vi, sprevi = usv_tree[i] 
        ur,sr, vr, sprevr = usv_tree[2*i+1]
        ul,sl, vl, sprevl = usv_tree[2*i+2]
        avec = np.kron(sr*ur, vr) + np.kron(sl*ul, vl) 
        diff = np.linalg.norm(vi-avec)
        if diff > 0.000001:
            print(i, diff)
            return -1
    return 0
def tree_with_nonzero_paths(avec, prob_threshold):
    ''' 
    generates Schmidt decomp tree
    returns a list of u, s,v, sprev and number of nonzeropath
    sprev: the prob along the path
    return usv_tree:
    a list of nodes
    the nodes in usv_tree: |0|1|2|3|4|...|N-2|
               0
              /\
             1 2
            /\ /\
           3 4 5 6   
    '''
    if avec.size <= 2: 
        return avec;  
    nonzeropath = 0
    #list of (u,s,v)
    usv_tree = [0]*(avec.size-1) #|0|1|2|3|4|...|N-2|

    usv_tree[0] = (1,1,avec,1)
    level = 0 
    u,s,v = find_childs(avec)

    #left child, sprev is the combination of schmidt coeff along the path
    sprev = 1;
    usv_tree[1] = (u[:,0],s[0],v[0], sprev*s[0])
    #right child, sprev is the combination of schmidt coeff along the path
    usv_tree[2] = (u[:,1],s[1],v[1], sprev*s[1])

    while(v.shape[1] > 2): #level child id
        level += 1
        istart = 2**(level)-1
        iend = istart+2**(level)
        inode = iend
        for i in range(istart, iend):
            # sprev is the combination of schmidt coefficients along the path
            sprev = 0
            if type(usv_tree[i]) != int:
                sprev = usv_tree[i][3]

            if  (sprev > prob_threshold): #if prob is too small, skip
                u,s, v = find_childs(usv_tree[i][2])
                #left child, 
                usv_tree[inode] = (u[:,0], s[0], v[0], sprev*s[0])
                #rightchild
                usv_tree[inode+1] = (u[:,1], s[1], v[1],sprev*s[1])
                if v.shape[1] == 2:
                    nonzeropath += 1
                print("svd node:{}, addnodes:{},{}"
                    .format(i, inode, inode+1))
            inode += 2 
            
    return usv_tree, nonzeropath



def sum_of_nonzeropaths(usv_tree, prob_threshold):
    '''
    remove paths that has the coefficient less than
    prob_threshold 
    '''
    tlen = len(usv_tree)
    sum_of_paths = 0
    path_vec = 0
    npaths = 0 #number of nonzero paths
    for i in range(int(tlen/2), tlen):
         if isinstance(usv_tree[i], tuple):
            ui, si, vi, sprevi = usv_tree[i]
            #TODO: we can combine nodes with the same parents
            if sprevi >= prob_threshold:
                path_vec = vi
                inode = i
                #go up from leaf node to the root
                while inode  > 0:
                    unode, snode, vnode, sprevnode = usv_tree[inode]
                    path_vec = np.kron(snode*unode, path_vec)
                    if inode%2 == 1:
                        inode = int(inode/2) # e.g. inode 7->3
                    else:
                        inode = int((inode-1)/2) #eg inode 8->3
                    #print("i and inode", i, inode)
                sum_of_paths += path_vec
                npaths += 1
    return sum_of_paths, npaths



if __name__ == "__main__" :
    
    n = 10 #taken as a matrix size
    N = 2**n
    threshold = 0.015
    psi = 0
    scase = 1;       filename = 'qft'
    n0path = 0
    Nsquare = N**2
    if scase == 1:#qft
        print('===============================================')
        print('case:', 'quantum fourier transform ')
        A = qft(N)
        a = A.flatten()
        psi = a 
    elif scase == 2:#random
        
        dist = "exponential";
        print('===============================================')
        print('case:', 'random '+dist)
        #rng = np.random.RandomState(0)
        rng = np.random.default_rng()
        X = 0
        if dist == "normal":
            X = rng.normal(size=(N,N))
        elif dist == "uniform":
            X = rng.uniform(size=(N,N))
        elif  dist == "exponential":
            X = rng.exponential( size=(N,N))
        elif  dist == "poisson":
            X = rng.poisson(size=(N,N))  

        psi = X.flatten()
        psi = psi/np.linalg.norm(psi)
        


    listofdecomps = np.array([3,N,N], dtype=complex)

    print('=================================================================\n\n')
    (usv_tree, nonzeropath) = generate_tree_elements(psi)


    '''
    the nodes on usv_tree: |0|1|2|3|4|...|14|empty]
                 0
                 /\
                 1 2
                /\ /\
               3 4 5 6       
    '''
    L = []
    for i in range(int(psi.size/2)-1, psi.size-1):
        if isinstance(usv_tree[i], tuple):
            u, s, v, sprev = usv_tree[i]
            print("shape:", v.shape)
            L.append(usv_tree[i][3])
        else:
            L.append(0)
    fig, ax = plt.subplots()
    ax.axvline(threshold, linestyle='--')
    values, bins, bars = ax.hist(L)

    ax.set_xlabel("probability (coefficient) of the path")
    ax.set_ylabel("Number of paths")
    ax.set_title('n: {}-qubits, distribution: {} '.format(int(np.log2(psi.size)), filename))
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
    #plt.savefig('{}{}qubits.eps'.format(filename,n ), bbox_inches='tight')
    #plt.savefig('{}{}qubits.png'.format(filename,n),bbox_inches='tight')
    
    """ listofprobs = []
    print('Recursive\n'\
        '=================================================================\n\n')
    n0path = recursive_find_childs(psi,1,0, listofprobs)
    print("n0path:{}".format(n0path))

    fig, ax = plt.subplots()
    values, bins, bars = ax.hist(listofprobs)

    ax.set_xlabel("probability (coefficient) of the path")
    ax.set_ylabel("Number of paths")
    ax.set_title('n:{}-qubits '.format(n))
    ax.bar_label(bars, fontsize=9, color='red') """