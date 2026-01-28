import numpy as np

# def sgd(
#         params,
#         lr: float = 0.001
# ) -> dict:
#     """
#     Implementation of the Stochastic Gradient Descent.

#     :param params: Parameters to optimize.
#     :type params: any
#     :param lr: Learning Rate.
#     :type lr: float
#     """
#     new_updates: dict   = {}
#     inner_updates: dict = {}

#     for layer in params:
#         for param in params[layer]:
#             if param == None:
#                 continue
#             value       = params[layer][param]["value"]
#             gradient    = params[layer][param]["gradient"]

#             res = value - (lr * gradient)

#             inner_updates[param] = {"gradient" : np.zeros(np.shape(value), np.dtype(value.dtype)), "value" : res}

#         new_updates[layer] = inner_updates
    
#     return new_updates


class SGD:
    def __init__(self, lr: float = 0.001):
        self.lr = lr
    
    def __call__(self, params, *args, **kwds):
        """
        Implementation of the Stochastic Gradient Descent.

        :param params: Parameters to optimize.
        :type params: any
        :param lr: Learning Rate.
        :type lr: float
        """
        new_updates: dict   = {}
        inner_updates: dict = {}

        for layer in params:
            for param in params[layer]:
                if param == None:
                    continue
                value       = params[layer][param]["value"]
                gradient    = params[layer][param]["gradient"]

                res = value - (self.lr * gradient)

                inner_updates[param] = {
                    "gradient"  : np.zeros(np.shape(value),np.dtype(value.dtype)),
                    "value"     : res
                }

            new_updates[layer] = inner_updates
        
        return new_updates