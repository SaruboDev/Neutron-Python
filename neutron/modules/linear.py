from neutron.core._module import Module, field
from dataclasses import dataclass

import numpy as np

class Linear(Module):
    weights: np.ndarray
    bias: np.ndarray
    trainable: bool = field(static = True)

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, dtype = np.float32) -> None:
        """
        Docstring for __init__
        
        :param input_dim: Input dimensions.
        :type input_dim: int
        :param output_dim: Output dimensions.
        :type output_dim: int
        :param use_bias: If the linear layer can add the bias to the output.
        :type use_bias: bool
        :param dtype: Dtype used
        """
        self.weights    = np.random.uniform(size = (output_dim, input_dim))
        self.bias       = np.random.uniform(size = (output_dim, )) if use_bias else None
        self.use_bias   = use_bias
        self.new_dtype  = np.dtype(dtype)
        self.trainable  = True

    def __call__(self, x):
        x = self.weights @ x
        if self.use_bias == True and self.bias is not None:
            x = x + self.bias
        
        # Not sure why it crashes without .name, but it works.
        if x.dtype != np.dtype(self.new_dtype.name):
            x = x.astype(self.new_dtype)

        return x
    
    def _get_layer_params(self) -> int:
        """
        Each layer type has it's own way of calculating parameters, for linear is (out * in) + out.
        """

        res = (self.weights.shape[0] * self.weights.shape[1])
        if self.bias != None:
            res += self.bias.shape[0]

        params = (res, 0) if self.trainable == True else (0, res)

        return params