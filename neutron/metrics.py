import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import pandas as pd

##############
### LOSSES ###
##############

class categorical_crossentropy:
    """
    The categorical crossentropy loss. Mostly used with One Hot Encoding data.
    Args:
    - from_logits: bool = If you plan to have your data already softmax-ed, turn this to False.
    """
    __name__ = "categorical_crossentropy"
    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def __call__(self, y_pred, y_true, *args, **kwds):
        if self.from_logits == True:
            logs = jax.nn.log_softmax(y_pred)
        else:
            logs = jnp.log(y_pred)

        res = - jnp.mean(jnp.sum(y_true * logs, axis = -1))
        return res

class sparse_categorical_crossentropy:
    """
    The sparse categorical crossentropy loss.
    Args:
    - from_logits: bool = If you plan to have your date already soft-maxed, turn this to false.
    - ignore_class: int|None = If you plan on having padded data, you can use this to set it to the corresponding int.
    """
    __name__ = "sparse_categorical_crossentropy"
    def __init__(self, from_logits: bool = True, ignore_class: int|None = None):
        self.from_logits    = from_logits
        self.ignore_class   = ignore_class
    
    def __call__(self, pred, true, *args, **kwds):
        if self.from_logits == True:
            # We get the log probabilities
            log_probs   = jax.nn.log_softmax(pred, axis = -1)
            # For compatibility we expand the dimension of the true values
            true_exp    = jnp.expand_dims(true, axis = -1)
            # We get the actual probability
            probs       = jnp.take_along_axis(log_probs, true_exp, axis = -1)
            # For efficiency we remove the last axis, so it's on par with the mask
            probs       = jnp.squeeze(probs, axis = -1)

            if self.ignore_class != None:
                # Now we get an array where 0 are the parts to ignore and 1 is the good stuff
                m   = jnp.where(true == self.ignore_class, 0, 1)

                print(f"m", m)
                # We multiply the negative probability with the mask
                lm  = (- probs) * m

                den = jnp.sum(lm)
                print(f"Den", den)

                # And just get our result
                res = jnp.sum(lm) / jnp.sum(m)
            else:
                # If the user did not set a custom class to ignore we just, you know, get the mean of the negative probs
                lm  = -probs
                res = jnp.mean(lm)
            return res

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

###################
### ACTIVATIONS ###
###################

class leaky_integrate_and_fire:
    """
    To be used with jax.lax.scan which requires a new carry to be returned.
    """
    __name__ = "LIF"
    v_reset: jnp.ndarray = eqx.field(static = True)
    thresh: jnp.ndarray
    leak: float = eqx.field(static = True)

    def __init__(self, v_reset, leak, init_thresh):
        self.v_reset = v_reset
        self.thresh = init_thresh
        self.leak = leak
        
    def __call__(self, v, input):
        res = self.leak * v + input
        s = (res >= self.thresh).astype(jnp.float32)
        
        res = jnp.where(s > 0, self.v_reset, res)
        return res, s