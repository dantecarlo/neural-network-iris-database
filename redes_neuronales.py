# %% Libraries
from IPython.display import clear_output
import time
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

from sklearn.datasets import make_circles

# %% Getting Iris
iris = datasets.load_iris()
X = iris.data[:, :3]
y = iris.target
y_c = np.empty([len(y), 1])

for i in range(len(y)):
    if(y[i] == 0):
        y_c[i] = [1.]
    elif(y[i] == 1):
        y_c[i] = [0.5]
    elif(y[i] == 2):
        y_c[i] = [0]

y_x = np.empty([len(y), 3])

for i in range(len(y)):
    if(y[i] == 0):
        y_x[i] = [0, 0, 1]
    elif(y[i] == 1):
        y_x[i] = [0, 1, 0]
    elif(y[i] == 2):
        y_x[i] = [1, 0, 0]

y = y_c


print(X)


# %% Plot

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()


plt.scatter(X[:, 0], X[:, 1], c=y_x, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# %% DataSet
# n: Numbers of registers
n = X.shape[0]

# p: Number of caracteristics per data
p = X.shape[1]

# X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)

# y = y[:, np.newaxis]

# plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c="skyblue")
# plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="salmon")
# plt.axis("equal")
# plt.show()

# %% Neural Layer


class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        # Function of activation
        self.act_f = act_f
        # Bias
        self.b = np.random.rand(1, n_neur) * 2 - 1
        # Weights
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1

# %% Activation Function


# [0] F(x), [1] der(F(x))
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

thip = (lambda x: (2 / (1 + np.e ** (-2*x))) - 1,
        lambda x: 1 - ((2 / (1 + np.e ** (-2*x))) - 1) ** 2)

relu = (lambda x: np.maximum(0, _x),
        lambda x: np.maximum(0, _x))

_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x))

# %% Neural Networl


def create_nn(topology, act_f):
    nn = []
    # Create layers
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))
    return nn


# Cost Function (error media cuadratic)
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))

# %% Training
# lr: learning rate


def train(neural_net, X, y, l2_cost, lr=0.5, train=True):

    out = [(None, X)]
    # if(train == False):

    # Forward pass
    for l, layer in enumerate(neural_net):

        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)

        out.append((z, a))

    if train:

        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):

            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(neural_net) - 1:
                deltas.insert(0, l2_cost[1](a, y) * neural_net[l].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            # Gradient descent
            neural_net[l].b = neural_net[l].b - \
                np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]


topology = [p, 4, 8, 1]

# neural_net = create_nn(topology, sigm)
# train(neural_net, X, y, l2_cost, 0.5)
# %% Trainning NN and Visualization


topology = [p, 8, 4, 1]

neural_n = create_nn(topology, sigm)

loss = []

for i in range(1000):
    # Training Network
    pY = train(neural_n, X, y, l2_cost, lr=0.05)


# %% Test
print(train(neural_n, np.array(
    [[6.7, 3.1, 4.7]]), y, l2_cost, train=False)[0][0])
