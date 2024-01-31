print("\n\n" + "=" * 30 + " JAX GPU TEST " + "=" * 30)

import jax
from jax.lib import xla_bridge
import jax.numpy as jnp

print("jax.__version__:", jax.__version__)
print("xla_bridge.get_backend().platform:", xla_bridge.get_backend().platform)
print("jax.devices():", jax.devices())
print("jax.default_backend():", jax.default_backend())

key = jax.random.PRNGKey(42)
x = jax.random.uniform(key, (5,))
print("x:", x)
y = jnp.ones_like(x)
print("y:", y)
print("x + y:", x + y)


import jaxlib
import flax 
import optax 
import distrax 
import chex
# import orbax

print("jaxlib.__version__:", jaxlib.__version__)
print("flax.__version__:", flax.__version__)
print("optax.__version__:", optax.__version__)
print("distrax.__version__:", distrax.__version__)
print("chex.__version__:", chex.__version__)

print("=" * 70 + "\n")


print("\n\n" + "=" * 30 + " Tensorflow GPU TEST " + "=" * 30)


import tensorflow as tf


print("tf.__version__:", tf.__version__)

print("tf.test.is_gpu_available():", tf.test.is_gpu_available(), "\n")

print("tf.test.is_gpu_available(cuda_only=True):", tf.test.is_gpu_available(cuda_only=True), "\n")

print("tf.config.list_physical_devices('\GPU\'):", tf.config.list_physical_devices('GPU'), "\n")


with tf.device('gpu:0'):
    x = tf.random.uniform((5, 1))
    print("x:", x)
    y = tf.ones_like(x)
    print("y:", y)
    print("x + y:", x + y)

print("=" * 70 + "\n")


# print("\n\n" + "=" * 30 + " Pytorch GPU TEST " + "=" * 30)

# import torch

# print("torch.__version__:", torch.__version__)
# print("torch.cuda.is_available():", torch.cuda.is_available())

# x = torch.rand(5).cuda()
# y = torch.ones_like(x).cuda()
# print("x:", x)
# print("y:", y)
# print("x + y:", x + y)

# print("=" * 70 + "\n")