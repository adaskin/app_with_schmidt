

"""
Created on Fri Dec  3 09:10:44 2021

@author: adaskin
"""
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from sim_tree import generate_tree_elements,sum_of_nonzeropaths
import sklearn.datasets as datasets
import scipy.sparse

def polar2xy(r, theta):
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    return x,y
  
def randPolarDot(r1, r2):
    r = random.random() # [0, 1]
    r = r*(r2-r1) + r1
    theta = random.random()*math.pi*2 #hatali
    return r, theta
figure, ax1 = plt.subplots()
#ax2 = figure2.add_subplot(2,2,1,projection='3d')

CL = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
nrings = 4
ndots = 256

ringSize = 5; 

X = np.zeros([nrings*ndots,2])
count_dot = 0 
for iring in range(nrings):
    for jdot in range(ndots):
        
        [r, theta] = randPolarDot(iring*ringSize, (iring+1)*ringSize) 
        [x, y] = polar2xy(r, theta) 
        X[count_dot,:] = [x, y]
        count_dot += 1
        
        ax1.scatter(x, y, color=CL[iring],marker='.')
        #ax2.scatter3D(x, y, x**2+y**2, c=CL[iring]);
        
#ax1.set_xlabel("x")
#ax1.set_ylabel("y")
figure.savefig('rings.eps', bbox_inches='tight')
figure.savefig('rings.png',bbox_inches='tight')

threshold = 0.020
    
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

ax.set_xlabel("probability (coefficient) of the path")
ax.set_ylabel("Number of paths")
ax.set_title('n: {}-qubits, data: {} '.format(int(np.log2(psi.size)), 'rings'))
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
plt.savefig('rings20qubits.eps', bbox_inches='tight')
plt.savefig('rings20qubits.png',bbox_inches='tight')
