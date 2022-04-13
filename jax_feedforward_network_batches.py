import jax
import jax.numpy as jnp
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
from jax import random
import torch
import numpy as np


def one_hot(x, length=10):
    '''
    convert MNIST label to one hot vectors
    '''
    vec = jnp.zeros((1,length), dtype=jnp.float32)
    label = vec.at[:,x-1].set(1.)  # set one hot to 1 at correct location for each sample
    return label

def softmax(a):
    '''
    takes output layer activations for an entire batch and applies a softmax for each sample
    '''
    exp_a =  jnp.exp(a)
    norm = jnp.sum(jnp.exp(a), axis=1, keepdims=1)  # do not sum over batch dim
    return exp_a / jnp.repeat(norm, a.shape[1], axis=1)

def crossentropy(y_pred, y_true):
    '''
    compute crossentropy, does not cover the edge case where prediction for one of the classes is exactly 0. TODO
    '''
    return - jnp.sum(y_true * jnp.log(y_pred))

def forward(weights, biases, x):
    return x @ weights + biases

def compute_loss(weights, biases, x, y, aggregation='sum'):
    '''
    compute loss over batch, aggregation determines how the loss is aggregated over the batch dimension.
    Options are sum or mean
    '''
    y_pred = forward(weights, biases, x)
    probs = softmax(y_pred)
    loss = crossentropy(probs, y)
    return loss

backward = jax.jit(jax.grad(compute_loss, argnums=(0,1)))

def train(data, weights, biases, niter, lr, batchsize):
    losses = list()
    x,y = data
    ndata = len(y)

    # epochs
    for i in range(niter):
        idx = np.array(range(ndata))
        np.random.shuffle(idx)
        split_idx = np.array(range(batchsize, ndata, batchsize))
        batch_idx = np.split(idx, split_idx)
        
        # batches
        for sample_idx in tqdm(batch_idx):
            xbatch, ybatch = x[sample_idx], y[sample_idx]
            loss = compute_loss(weights, biases, xbatch, ybatch, aggregation='mean')
            losses.append(loss)
            dw,db = backward(weights, biases, xbatch,ybatch)
            weights = weights - lr * dw
            biases = biases - lr * db
    return weights, biases, losses


### START EXPRIMENT ###

# random initialization of network
key = random.PRNGKey(0)
weights = random.normal(key, (784,10))
key, subkey = random.split(key)
biases = random.uniform(subkey, (1,10))

print("Preprocessing...")
mnist_data = MNIST('/tmp/mnist/', download=True)
ndata = len(mnist_data)
y = np.zeros((ndata, 10))
ims = np.zeros((ndata, 784))
for i, dp in tqdm(enumerate(mnist_data)):
    im,label = dp
    ims[i] = np.reshape((np.array(im)/255), (1,-1))
    y[i] = one_hot(label)

data = (ims, y)

# train
print("training ...")
w,b,l = train(data, weights, biases, 100, 0.001, 10)

plt.figure()
plt.plot(l)
plt.show()



# debug
xb = ims[:20]
yb = y[:20]
compute_loss(weights, biases, xb,yb)

y_pred = forward(weights, biases, xb)
loss = crossentropy(y_pred, yb)




