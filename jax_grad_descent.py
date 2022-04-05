import jax
import jax.numpy as jnp

def error(y_pred, y_true):
    return (y_pred - y_true)**2

error_dy = jax.grad(error)

nsteps = 10000
lr = 0.001
x = 10.

for i in range(nsteps):
    grad = error_dy(x, 0.)
    x = x - lr*grad

print(x)
