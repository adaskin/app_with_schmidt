import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sim_tree import generate_tree_elements,sum_of_nonzeropaths

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1_000, test_size=256
)

#test_norm = np.linalg.norm(X_test.flatten())
#X_test = X_test/(test_norm)
rng = np.random.RandomState(0)
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise

noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise

import matplotlib.pyplot as plt


def plot_digits(X, title):
    """Small helper function to plot 100 digits."""
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)




A = np.zeros([256,256])
#A[0:100,:] = X_test_noisy
x = X_test_noisy.flatten()
#a = A.flatten()
#a[0:25600] = x
psi = x

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

values, bins, bars = ax.hist(L)

ax.set_xlabel("probability (coefficient) of the path")
ax.set_ylabel("Number of paths")
ax.set_title('n:{}-qubits '.format(8))
ax.bar_label(bars, fontsize=9, color='red')


p, npaths = sum_of_nonzeropaths(usv_tree,1)
print(np.linalg.norm(p-psi))
print(np.linalg.norm(p-psi,1))
print(np.dot(psi,p))
X_tensor = p.reshape(256,256)
plot_digits(
   X_test, f"Noisy test images\nMSE: {np.mean((X_test - X_tensor) ** 2):.2f}"
)

plot_digits(
   X_tensor, f"Noisy test images\nMSE: {np.mean((X_test - X_tensor) ** 2):.2f}"
)

