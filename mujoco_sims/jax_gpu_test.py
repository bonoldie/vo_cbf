import jax
import jax.numpy as jnp

print(jax.devices())

gpu = jax.devices("gpu")[0]

def square(x):
    return x * x

square_gpu = jax.jit(square)

x = jax.device_put(jnp.array(4.0), gpu)
result = square_gpu(x)

# Force completion, because JAX execution is asynchronous.
result.block_until_ready()

print(result)
print(result.device)