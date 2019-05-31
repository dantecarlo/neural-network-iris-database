# %% Libraries
from IPython.display import clear_output
import time
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

from sklearn.datasets import make_circles


# %% Activation Functions


sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

tanh = (lambda x: (2 / (1 + np.e ** (-2*x))) - 1,
        lambda x: 1 - ((2 / (1 + np.e ** (-2*x))) - 1) ** 2)

relu = (lambda x: np.maximum(0, _x),
        lambda x: np.maximum(0, _x))

_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x))

# %% Cost Function

cost_fun = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
            lambda Yp, Yr: (Yp - Yr))


# %% DataSet
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

# %% Plot Dataset

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


# %% Class Neural Network


class NeuralNetwork:
    def __init__(self, X, y, act_fun, cost_fun, topology):
        self.X = X
        self.y = y
        self.act_fun = act_fun
        self.cost_fun = cost_fun
        self.topology = topology
        self.neural_net = []

    class neural_layer():
        def __init__(self, n_conn, n_neur, act_fun):
            # Activation Function
            self.act_fun = act_fun
            # Bias
            self.b = np.random.rand(1, n_neur) * 2 - 1
            # Weights
            self.W = np.random.rand(n_conn, n_neur) * 2 - 1

    def create_nn(self):
        # Create layers
        # topology[l] -> number of input connection
        # topology[l+1] -> number of neurons
        for l, layer in enumerate(self.topology[:-1]):
            self.neural_net.append(self.neural_layer(
                self.topology[l], self.topology[l+1], self.act_fun))

    # lr: learning rate it: iterations
    def subtrain(self, X, lr=0.05, train=True):
        self.lr = lr

        # out -> Output of each layer
        if type(X) == type(self.X):
            out = [(None, self.X)]
        else:
            out = [(None, X)]

        # Forward pass
        for l, layer in enumerate(self.neural_net):
            # z -> Weighted sum
            z = out[-1][1] @ self.neural_net[l].W + self.neural_net[l].b
            # a -> Activation (exit of the layer)
            a = self.neural_net[l].act_fun[0](z)
            out.append((z, a))

        if train:
            # Backward pass
            deltas = []
            for l in reversed(range(0, len(self.neural_net))):

                z = out[l+1][0]
                a = out[l+1][1]
                # Last Layer
                if l == len(self.neural_net) - 1:
                    # Delta in last layer
                    deltas.insert(0, self.cost_fun[1](
                        a, self.y) * self.neural_net[l].act_fun[1](a))
                else:
                    # Delta of previous layer
                    deltas.insert(0, deltas[0] @ _W.T *
                                  self.neural_net[l].act_fun[1](a))

                _W = self.neural_net[l].W

                # Gradient descent
                self.neural_net[l].b = self.neural_net[l].b - \
                    np.mean(deltas[0], axis=0, keepdims=True) * self.lr
                self.neural_net[l].W = self.neural_net[l].W - \
                    out[l][1].T @ deltas[0] * self.lr

        return out[-1][1]

    # Training Network
    def train(self, it=100000):
        self.create_nn()
        self.it = it

        for i in range(self.it):
            self.subtrain(self.X)

    def analize(self, input):
        return self.subtrain(input, self.lr, train=False)[0][0]

# %% Train and Test


# p: Number of caracteristics per data
p = X.shape[1]

topology = [p, 8, 4, 1]

nn = NeuralNetwork(X, y, sigm, cost_fun, topology)

nn.train(1000)

print(nn.analize([6.7, 3.1, 4.7]))
