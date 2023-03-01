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

#data= datasets.fetch_20newsgroups_vectorized()
#data= datasets.load_iris()
#data= datasets.load_digits()
data, y = datasets.make_circles(n_samples=1024, factor=0.3, noise=0.05, random_state=0)

filename = '20newsgroup'#data.filename.split('.')[0]

X = data.data[0:1024,:]
n = 8
N = 2**n

threshold = 0.5
    
G = X@X.transpose()
if scipy.sparse.issparse(G):
    G = G.toarray()
    
psi = G.flatten()
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
fig, ax = plt.subplots()
ax.axvline(threshold, linestyle='--')
values, bins, bars = ax.hist(L)

ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
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
#plt.savefig('{}{}qubits.eps'.format(filename,n ), bbox_inches='tight')
#plt.savefig('{}{}qubits.png'.format(filename,n),bbox_inches='tight')
