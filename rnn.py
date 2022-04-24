
'''
implementation of a recurrent network predicting next digits in a sequence
reference implementation: https://roberttlange.github.io/posts/2020/03/blog-post-10/
'''

from tqdm import tqdm
import matplotlib.pyplot as plt 
import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
from jax import lax

'''
define a recurrent neural network to predict all sequences of length 3 on the ring N % 10
'''

X = jnp.array([0., 1., 2., 3., 4. ,5., 6., 7.,8., 9.])  # possible input digits
seqlen = 5
X = jnp.expand_dims(X,1)
X = jnp.repeat(X,seqlen,axis=1)

key = random.PRNGKey(0)
weights_ff_l1 = random.normal(key, (1,32))
key, subkey = random.split(key)
weights_rc_l1 = random.normal(subkey, (32,32))
key, subkey = random.split(key)
biases_l1 = random.uniform(subkey, (1,32))
key, subkey = random.split(key)
weights_l2 = random.normal(subkey, (32,10))
key, subkey = random.split(key)
biases_l2 = random.normal(subkey, (1,10))

params = [
    [weights_ff_l1, weights_rc_l1, biases_l1], 
    [weights_l2, biases_l2]
]

def recurrent_layer(params, x, u):
    def step(params, u, x):
        a_t_forward = x @ params[0]
        a_t_recurrent = u @ params[1]
        a_t = a_t_forward + a_t_recurrent + params[2]  # sum and add bias
        h_t = jnp.tanh(a_t)
        return h_t, h_t

    f = partial(step, params)
    x_t = jnp.moveaxis(x, 1, 0)  # time dimension must be first for scanning across time
    _, out = lax.scan(f, u, x_t)
    x_batch = jnp.moveaxis(x,0,1)  # move batch dim back to front
    return out


def output_layer(params, x):
    return (x @ params[0]) + params[1]

output_layer = jax.vmap(output_layer, in_axes=(None,0))

def softmax(x):
    m = jnp.max(x)
    return (jnp.exp(x - m)) / jnp.sum(jnp.exp(x - m))

batch_softmax = jax.vmap(softmax, in_axes=(0,))

def forward(params, xs):
    hidden_params = params[0]
    output_params = params[1]
    
    hidden_dim = hidden_params[2].shape[1]
    u_init = jnp.zeros((1,hidden_dim))

    hidden_states = recurrent_layer(hidden_params, xs, u_init)
    a_out = output_layer(output_params, hidden_states)
    return a_out

forward_batch = jax.jit(jax.vmap(forward, in_axes=(None,0)))

def neg_crossentropy(x, labels):
    return - jnp.sum(jnp.log(x) * labels)

def logit_neg_crossentropy(x, labels):
    m = jnp.max(x)
    log_probs = (x - m) / jnp.sum(jnp.exp(x - m))
    return - jnp.sum(log_probs * labels)

def compute_training_loss(params, x, labels):
    logits = forward(params, x)
    loss = logit_neg_crossentropy(logits, labels)
    return loss

backward = jax.jit(jax.value_and_grad(compute_training_loss))

compute_training_loss_batch = jax.jit(jax.vmap(compute_training_loss, in_axes=(None,0,0)))

def compute_mean_training_loss(params, x, labels):
    losses = compute_training_loss_batch(params, x, labels)
    return jnp.mean(losses)

backward_batch = jax.jit(jax.value_and_grad(compute_mean_training_loss))

def optimize_SGD(params, grads, lr):
    '''
    recursively traverse list of parameters and associated gradients
    until a DeviceArray is encountered, then applies SGD update.
    returns a new nested list of parameters after applying the gradients.
    '''
    if not (type(params) == list):
        return params - (lr * grads)
    else:
        new_params = []
        for p,g in zip(params, grads):
            new_params.append(optimize_SGD(p,g,lr))
        return new_params

def train_step(params, x, labels, lr):
    loss, grads = backward(params, x, labels)
    new_params = optimize_SGD(params, grads, lr)
    return loss, new_params

def train_step_batch(params, x, labels, lr):
    loss, grads = backward_batch(params, x, labels)
    new_params = optimize_SGD(params, grads, lr)
    return loss, new_params

def one_hot(label, nclasses):
    one_hot_labels = jnp.zeros((nclasses,))
    one_hot_labels = one_hot_labels.at[label].set(1.)
    return one_hot_labels

batch_one_hot = jax.vmap(one_hot, in_axes=(0,None))

def predict(params, x):
    logits = forward(params, x)
    print(logits.shape)
    return jnp.argmax(logits, axis=2)


# train
add = jnp.array(range(1,seqlen+1))
losses = []
epochs = 1000000
labels = X + add
X = jnp.expand_dims(X,1)

oh_labels = [batch_one_hot(l.astype(int), 10) for l in labels]
oh_labels = jnp.stack(oh_labels)

for i in tqdm(range(epochs)):
        loss, params = train_step_batch(params, X, oh_labels, 0.001)
        losses.append(loss)


plt.figure()
plt.plot(losses)
plt.xlabel("step")
plt.ylabel("loss")

print(predict(params, jnp.array([[1,1,1,1]])))
print(softmax(forward(params, jnp.array([[1,1,1]]))))
plt.show()


