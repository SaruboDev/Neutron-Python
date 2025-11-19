import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import pandas as pd

###############
### METRICS ###
###############

def accuracy(logits, y_true, *args, **kwargs):
    """
    Calculates the accuracy given the logits and the true labels.
    """
    if logits.ndim > 1:
        # This is for ohe.
        acc = jnp.mean(jnp.argmax(logits, axis = -1) == jnp.argmax(y_true, axis = -1))

    elif logits.ndim == 1:
        # This is for binary.
        acc = jnp.mean(jnp.argmax((jax.nn.sigmoid(logits) >= 0.5).astype(jnp.int32), axis = -1) == y_true)

    return acc
