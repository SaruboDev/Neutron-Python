from neutron.activations import softmax
from neutron.core._tracer import Tracer

import numpy as np

class log_loss:
    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits

    def __call__(self, y_pred, y_true):
        """
        Applies the log loss, or Cross Entropy to calculate your model loss.\n
        NOTE: For the logit formula, i think there's a way more optimized version without using softmax.

        :param y_true: The correct label.
        :param y_pred: The predicted output.
        :param from_logits: True if you want to use raw numbers instead of probability.
        :type from_logits: bool
        :return: Loss
        :rtype: float
        """ 
        result: Tracer

        if not self.from_logits:
            result = -np.sum((y_true * np.log(y_pred + 1e-12)), axis = 1, keepdims = False)
        else:
            result = np.sum(-(y_true * np.log(softmax(y_pred - np.max(y_pred, axis = 1, keepdims = True)))), axis = 1, keepdims = False)

        return result
