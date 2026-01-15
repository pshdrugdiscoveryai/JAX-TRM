"""
Implementing each layer to called in the final TRM model
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


def _find_multiple(a, b):
    return (-(a // -b)) * b


def trunc_normal_init(key, shape, std=1.0):
    """Truncated normal initialization for JAX."""
    # Sample from normal distribution
    values = jax.random.normal(key, shape) * std
    # Truncate to [-2*std, 2*std]
    return jnp.clip(values, -2 * std, 2 * std)


class CastedLinear(nn.Module):
    in_features: int
    out_features: int
    bias: bool = False
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Truncated LeCun normal init
        std = 1.0 / (self.in_features ** 0.5)
        self.kernel_init = lambda key, shape: trunc_normal_init(key, shape, std=std)
        
        if self.bias:
            self.bias_init = nn.initializers.zeros
        else:
            self.bias_init = None
    
    @nn.compact
    def __call__(self, x):
        # Cast input dtype for computation
        compute_dtype = self.dtype
        
        kernel = self.param('kernel', 
                           self.kernel_init, 
                           (self.in_features, self.out_features))
        
        # Cast kernel to match input dtype
        kernel = kernel.astype(x.dtype)
        
        y = jnp.dot(x, kernel)
        
        if self.bias:
            bias = self.param('bias', self.bias_init, (self.out_features,))
            bias = bias.astype(x.dtype)
            y = y + bias
            
        return y


class SwiGLU(nn.Module):
    hidden_size: int
    expansion: float = 8/3  # Common default
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        inter = _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
        
        self.gate_up_proj = CastedLinear(
            in_features=self.hidden_size,
            out_features=inter * 2,
            bias=False,
            dtype=self.dtype
        )
        self.down_proj = CastedLinear(
            in_features=inter,
            out_features=self.hidden_size,
            bias=False,
            dtype=self.dtype
        )
    
    def __call__(self, x):
        # Project to gate and up
        gate_up = self.gate_up_proj(x)
        
        # Split into gate and up projections
        gate, up = jnp.split(gate_up, 2, axis=-1)
        
        # Apply SwiGLU: silu(gate) * up, then project down
        return self.down_proj(nn.silu(gate) * up)
    
if __name__ == "__main__":
    # Initialize
    key = jax.random.PRNGKey(0)
    model = SwiGLU(hidden_size=512, expansion=8/3)

    # Dummy input
    x = jnp.ones((2, 128, 512))  # [batch, seq_len, hidden_size]

    # Initialize parameters
    variables = model.init(key, x)

    # Forward pass
    output = model.apply(variables, x)
    print(output.shape)  # (2, 128, 512)