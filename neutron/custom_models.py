import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
import math
from typing import Callable

class MLP(eqx.Module):
    """
    A simple MLP implementation. Not really usable for complex stuff, honestly it's here only for test purposes, but maybe
    could be useful for learning.
    """
    n_hidden: int           = eqx.field(static = True)
    activation: Callable    = eqx.field(static = True)
    dropout_rate: float     = eqx.field(static = True)
    vocab_size: int         = eqx.field(static = True)
    embedding_dim: int      = eqx.field(static = True)
    n_outputs: int          = eqx.field(static = True)
    outActivation: Callable = eqx.field(static = True)
    layers: list
    embed   : eqx.nn.Embedding
    dropout : eqx.nn.Dropout
    output  : eqx.nn.Linear

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            n_hidden: int,
            n_outputs: int,
            activation,
            outActivation = None,
            dropout_rate: float = 0.5,
            key: jr.PRNGKey = jr.PRNGKey(42)
    ):
        """
        Initializes the simple MLP model.
        Args:
        - vocab_size: int
        - embedding_dim: int
        - n_hidden: int = How many Linear layers you want in the model.
        - n_outputs: int = How many outputs should the model return.
        - activation = Which activation the model should use between Linear layers.
        - outActivation = Which activation the model should use for the output.
        - dropout_rate: float = Float for the dropout before the output layer.
        """
        keyEmbed, keyOut, keys = jr.split(key, 3)
        keys    = jr.split(keys, n_hidden)

        self.n_hidden       = n_hidden
        self.activation     = activation
        self.outActivation  = outActivation
        self.dropout_rate   = dropout_rate
        self.embedding_dim  = embedding_dim
        self.vocab_size     = vocab_size
        self.n_outputs      = n_outputs

        self.embed      = eqx.nn.Embedding(vocab_size, embedding_dim, key = keyEmbed)
        self.layers     = [eqx.nn.Linear(embedding_dim, embedding_dim, key = keys[i]) for i in range(n_hidden)]
        self.dropout    = eqx.nn.Dropout(p = dropout_rate, inference = False)
        self.output     = eqx.nn.Linear(embedding_dim, n_outputs, key = keyOut)
    
    def __call__(self, x, key: jr.PRNGKey):
        x   = jax.vmap(self.embed)(x)

        xLayers = x

        for layer in self.layers:
            xLayers     = jax.vmap(layer)(xLayers)
            xLayers     = self.activation(xLayers)

        # x       = jnp.transpose(xLayers, (1, 0))
        x = jnp.mean(xLayers, axis=0)
        drop    = self.dropout(x, key = key)
        output  = self.output(drop)

        if self.outActivation != None:
            output  = self.outActivation(output)

        return output


class FeedForward(eqx.Module):
    """
    A simple Feed Forward Network. Without Dropout.\n
    Initialize the layer by calling::
    
        ffn = FeedForward(embedding_dim, mid_dim, key = key)
    
    If you don't include the key inside init it will crash.\n
    The Feed Forward Layer is basically just a really quick and easy MLP with 2 linear layers, one with a non-linear activation (like ReLU),
    and another with a linear activation (or none).\n
    Some may prefer adding more layers along these 2 usual ones, like Dropouts.
    """
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, embedding_dim: int, mid_dim: int = None, key: jr.PRNGKey = None):
        """
        Initializes the Feed Forward layer.
        Args \n
        **embedding_dim**: int = Embedding dimension (or d_model) as the input for the first layer and output for the second layer.\n
        **mid_dim**: int = Embedding dimension (defaults as embedding_dim) as the output for the first layer and input for the second layer.\n
        **key**: jax.random.PRNGKey = A random PRNGKey, it's None, but will crash if you don't give a value. (Should be handled by itself if you connect it correctly though).
        """
        key1, key2 = jr.split(key, 2)
        self.linear1 = eqx.nn.Linear(in_features = embedding_dim, out_features = mid_dim, key = key1)
        self.linear2 = eqx.nn.Linear(in_features = mid_dim, out_features = embedding_dim, key = key2)
    
    def __call__(self, x):
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.linear2)(x)
        
        return x

class PositionalEncoding(eqx.Module):
    """
    A simple Positional Encoding layer.\n
    Initialize the layer by calling::
    
        pe = PositionalEncoding(embedding_dim, max_len)

    Positional Encoding is a technique where you give data inside the transformer a position vector to make the model able to
    understand it's connection with adjacent data.

    For example:\n
    "Hi, my name is Ron"\n
    Where the bytes would be:\n
    [b"H", b"i", b",", b" ", b"m", b"y", b" ", b"n", b"a", b"m", b"e", b" ", b"i", b"s", b" ", b"R", b"o", b"n"]\n
    The positional encoding makes the model read their position like (not a true representation of how it works):\n
    [1: b"H", 2: b"i", 3: b",", 4: b" ", 5: b"m", 6: b"y", 7: b" ", 8: b"n", 9: b"a", 10: b"m", 11: b"e", 12: b" ", 13: b"i", 14:b"s", 
    15: b" ", 16: b"R", 17: b"o", 18: b"n"]\n
    This is needed because transformer models read all data given in parallel (or almost), unlike RNN models where they read data sequentially.
    """
    embedding_dim: int = eqx.field(static = True)
    max_len: int = eqx.field(static = True)
    pe: jnp.ndarray = eqx.field(static = True)

    def __init__(self, embedding_dim: int, max_len: int):
        """
        A simple Positional Encoding layer.\n
        Args\n
        **embedding_dim**: int = Embedding dimension (or d_model)\n
        **max_len**: int = The max length of your sequence, if the sequence if longer than max_len, the extra things will be discarded.
        """
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        pe = jnp.zeros((max_len, embedding_dim))
        position = jnp.arange(0, max_len, dtype = jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[None]
        self.pe = jax.device_put(pe)
    
    def __call__(self, x):
        if x.ndim == 2:
            pe = jnp.squeeze(self.pe, axis = 0)
        elif x.ndim == 3:
            pe = self.pe
        x = x + pe[:, :x.shape[1]]
        return x

