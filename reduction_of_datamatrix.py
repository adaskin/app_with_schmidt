#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:00:15 2023

@author: adaskin
"""

import numpy as np
import matplotlib.pyplot as plt
from sim_tree import generate_tree_elements,sum_of_nonzeropaths

plt.rcParams['text.usetex'] = True

n = 10
N = 2**n

dist = "normal"
threshold = 0.4
filename = 'randX'+dist
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
ax.set_xlabel("Probability (coefficient) of the path")
ax.set_ylabel("Number of paths")
ax.set_title('n:{}-qubits, distribution:{} '.format(int(np.log2(psi.size)), dist))
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

plt.savefig('{}{}qubits.eps'.format(filename,n ), bbox_inches='tight')
plt.savefig('{}{}qubits.png'.format(filename,n),bbox_inches='tight')
