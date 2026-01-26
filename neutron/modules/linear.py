from neutron.core._module import Module
from dataclasses import dataclass

import numpy as np

@dataclass
class Linear(Module):
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, dtype = np.float64):
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
        self.dtype      = dtype

    def __call__(self, x):
        x = self.weights @ x
        if self.use_bias == True and self.bias is not None:
            x = x + self.bias

        if x.value.dtype != self.dtype:
            x.value.astype(self.dtype)

        return x