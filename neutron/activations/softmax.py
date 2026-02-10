import numpy as np

def softmax(x: np.ndarray):
    """
    Applies the softmax to convert raw numbers into a probability.
    
    :param x: Input array
    :type x: np.ndarray
    """

    x_exp = np.exp(x - np.max(x, axis = 1, keepdims = True))
    result = x_exp / (np.sum(x_exp, axis = 1, keepdims = True))
    return result