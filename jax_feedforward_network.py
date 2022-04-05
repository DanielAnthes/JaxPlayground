import jax
from jax import random
import jax.numpy as jnp
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm

mnist_dataset = MNIST('/tmp/mnist/', download=True)

def one_hot(x, length=10):
    vec = jnp.zeros((1,length), dtype=jnp.float32)
    label = vec.at[x-1].set(1.)
    return label

key = random.PRNGKey(0)
weights = random.normal(key, (784,10))
key, subkey = random.split(key)
biases = random.uniform(subkey, (1,10))

def softmax(a):
    return jnp.exp(a) / (jnp.sum(jnp.exp(a)))

def crossentropy(y_pred, y_true):
    return - jnp.sum(y_true * jnp.log(y_pred))

def forward(weights, biases, x):
    a = x @ weights + biases
    return softmax(a)

def compute_loss(weights, biases, x, y):
    y_pred = forward(weights,biases, x)
    loss = crossentropy(y_pred, y)
    return loss

backward = jax.jit(jax.grad(compute_loss, argnums=(0,1)))

def train(mnist_dataset, weights, biases, niter, lr):
    losses = list()
    for i in tqdm(range(niter)):
        for im, label in mnist_dataset:
            x = jnp.reshape((jnp.array(im)/255), (1,-1))
            y = one_hot(label)
            loss = compute_loss(weights, biases, x, y)
            losses.append(loss)
            dw,db = backward(weights, biases, x,y)
            weights = weights - lr * dw
            biases = biases - lr * db
    return weights, biases, losses

w,b,l = train(mnist_dataset, weights, biases, 10, 0.01)

plt.figure()
plt.plot(l)
plt.show()
