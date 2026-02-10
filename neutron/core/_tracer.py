import numpy as np

##############
### Tracer ###
##############

class Tracer:
    """
    **Tracer** is the object that keeps track of every operation done inside an AI Model.\n
    In **jax** it can be seen as 'jax.interpreters. something something. Tracer'.\n
    In **pytorch** it's what you'd see from the Variable (or Tensor).\n
    During the forward process, every operation done calls it's respective function, where a new Tracer with the output of the
    operation is created and has it's parents set as the two operands with the operator to know which derivatives to calculate.

    :TODO: May need to add Numpy compat with __array_ufunc__ and __array_function__, also reverse dunder for right hand ops.
    """
    value: int|float|np.ndarray = 0
    parents_operations: list = []

    gradient = 0.0

    def __init__(self, value) -> None:
        self.value = value
        self.parents_operations = []
        self.gradient = 0.0

    def __setparentop__(self, operation: list) -> None:
        """
        Sets the current Tracer parents, and which operation they did.
        
        :param operation: A list for the operations needed to calculate the current Tracer. [parent1, back_op_callable, parent2].
        :type operation: list
        """
        self.parents_operations = operation

    def __repr__(self) -> str:
        """
        Representation of the Tracer.
        """
        if type(self.value) == np.ndarray:
            return f"{self.value.dtype}[{np.shape(self.value)}]"
        
        return f"{type(self.value)}[{self.value}]"
    
    def __getattr__(self, name):
        return getattr(self.value, name)

    def __array__(self):
        return self.value
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        inputs = [x.value if isinstance(x, Tracer) else x for x in args]
        result = getattr(ufunc, method)(*inputs, **kwargs)

        new_tracer  = Tracer(result)
        all_parents = [x for x in args if isinstance(x, Tracer)]

        # parent_1 = all_parents[0] if len(all_parents) > 0 else None
        # parent_2 = all_parents[1] if len(all_parents) > 1 else None
        parent_1 = args[0] if len(args) > 0 else None
        parent_2 = args[1] if len(args) > 1 else None

        backwards_map = {
            "add"           : self.back_add,
            "subtract"      : self.back_sub,
            "multiply"      : self.back_mul,
            "log"           : self.back_log,
            "true_divide"   : self.back_truediv,
            "matmul"        : self.back_matmul,
            "exp"           : self.back_exp
        }
        operation = backwards_map.get(ufunc.__name__)

        new_tracer.__setparentop__([parent_1, operation, parent_2])
        return new_tracer

    def __array_function__(self, func, types, *args, **kwargs):
        backwards_map: dict = {
            "sum":  self.back_sum,
            "max":  self.back_max,
            "mean": self.back_mean
        }
        variables, named_args = args

        if func is np.sum:
            result = func(variables[0].value, **named_args)
        elif func is np.max:
            result = func(variables[0].value, **named_args)
        elif func is np.mean:
            result = func(variables[0].value, **named_args)
        else: 
            return NotImplemented
            
        operation = backwards_map.get(func.__name__)

        new_tracer  = Tracer(result)
        all_parents = [x for x in variables if isinstance(x, Tracer)]

        parent_1 = all_parents[0] if len(all_parents) > 0 else None
        parent_2 = all_parents[1] if len(all_parents) > 1 else None

        new_tracer.__setparentop__([parent_1, operation, parent_2, named_args])

        return new_tracer

    def __neg__(self):
        result = Tracer(-self.value)
        result.__setparentop__([self, self.back_neg, None])

        return result

    # Exceptions
    def astype(self, new_dtype) -> Tracer:
        """
        Changes dtype.
        """
        if type(self.value) == np.ndarray:
            output = Tracer(self.value.astype(np.dtype(new_dtype.name)))
            output.__setparentop__([self, self.back_dtype, None])
            return output
        else:
            raise Exception("You can't change dtype of a non array variable!")

    # Normal Operations
    def __add__(self, other) -> Tracer:
        """
        Addition
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value + other_value)

        output.__setparentop__([self, self.back_add, other])
        return output
    
    def __sub__(self, other) -> Tracer:
        """
        Subtraction
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value - other_value)

        output.__setparentop__([self, self.back_sub, other])
        return output
    
    def __mul__(self, other) -> Tracer:
        """
        Multiplication
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value * other_value)

        output.__setparentop__([self, self.back_mul, other])
        return output

    def __truediv__(self, other) -> Tracer:
        """
        Division
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value / other_value)

        output.__setparentop__([self, self.back_truediv, other])
        return output
    
    def __floordiv__(self, other) -> Tracer:
        """
        Int Division
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value // other_value)

        output.__setparentop__([self, self.back_floordiv, other])
        return output
    
    def __pow__(self, other) -> Tracer:
        """
        Exponentiation
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value ** other_value)

        output.__setparentop__([self, self.back_pow, other])
        return output
    
    def __mod__(self, other) -> Tracer:
        """
        Modulo
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value % other_value)

        output.__setparentop__([self, self.back_mod, other])
        return output
    
    def __matmul__(self, other) -> Tracer:
        """
        Matrix Multiplication
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(self.value @ other_value)

        output.__setparentop__([self, self.back_matmul, other])
        return output

    # Right hand operations
    def __radd__(self, other) -> Tracer:
        """
        R-Addition
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value + self.value)

        output.__setparentop__([other, self.back_add, self])
        return output
    
    def __rsub__(self, other) -> Tracer:
        """
        R-Subtraction
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value - self.value)

        output.__setparentop__([other, self.back_sub, self])
        return output
    
    def __rmul__(self, other) -> Tracer:
        """
        R-Multiplication
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value * self.value)

        output.__setparentop__([other, self.back_mul, self])
        return output

    def __rtruediv__(self, other) -> Tracer:
        """
        R-Division
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value / self.value)

        output.__setparentop__([other, self.back_truediv, self])
        return output
    
    def __rfloordiv__(self, other) -> Tracer:
        """
        R-Int Division
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value // self.value)

        output.__setparentop__([other, self.back_floordiv, self])
        return output
    
    def __rpow__(self, other) -> Tracer:
        """
        R-Exponentiation
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value ** self.value)

        output.__setparentop__([other, self.back_pow, self])
        return output
    
    def __rmod__(self, other) -> Tracer:
        """
        R-Modulo
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value % self.value)

        output.__setparentop__([other, self.back_mod, self])
        return output
    
    def __rmatmul__(self, other) -> Tracer:
        """
        R-Matrix Multiplication
        """
        other_value = other.value if isinstance(other, Tracer) else other

        output = Tracer(other_value @ self.value)

        output.__setparentop__([other, self.back_matmul, self])
        return output
    
    # Backwards
    def backwards(self, need_whole_graph: bool = False) -> None:
        """
        This method gets called ONLY for the last Tracer item obtained from a Model.
        Meaning that the backwards process throughout the relevant Tracers starts from here.
        """
        self.gradient   = np.ones_like(self.value) # Ones-like because it's the last value, so the one with gradient 1 by default.
        order           = reversed(topological_order(self))

        # dy/dx = dy/du * du/dx
        # Basically the y is the final loss to minimize, making dy/dx the gradient of x.
        # dy/du is the loss of the current item (self) with respect to the loss.
        # du/dx is the result of the basic derivative.

        # Will actually do the backwards process for each item from the last to the first, with respect to the priority order.
        # And inside the back_* functions, i did reshape the variables myself, but numpy would've broadcasted them itself.
        for tracer in order:
            if len(tracer.parents_operations) > 0 and tracer.parents_operations[1] is not None:
                a = tracer.parents_operations[0]
                b = tracer.parents_operations[2]

                arguments = None
                if len(tracer.parents_operations) > 3:
                    arguments = tracer.parents_operations[-1]
                # arguments = tracer.parents_operations[3] if len(tracer.parents_operations) == 3 else None
                tracer.parents_operations[1](tracer, a, b, arguments)
        
        if need_whole_graph == True:
            return
        
        self.parents_operations = None

        return

    def back_reshape(self, gradient, value, from_call:str):
        """
        This works by basically looping through both gradients shapes.
        Basically, if gradient has more dims compared to value, we just sum through the extra.
        Then, considering we got extra, we just check if the value (which most likely will have
        less dims than gradient) is 1 and if the gradient is > 1.
        Then just sum through those axis while keeping the same dimension as the value.
        """
        to_sum: list = [] # I'll save each dimension index
        len_gradient: int   = len(gradient.shape)
        len_value: int      = len(value.shape)

        for idx in range(len_gradient - len_value): # Check for extra dims
            to_sum.append(idx)

        for idx in range(len_value):
            offset = len_gradient - len_value
            if value.shape[idx] == 1 and gradient.shape[idx + offset] > 1:
                to_sum.append(idx + offset)

        if to_sum:
            gradient = np.sum(gradient, axis = tuple(to_sum), keepdims = True)
        
        return np.reshape(gradient, np.shape(value))

    def back_add(self, prev_node, a, b, args) -> None:
        # ADD : f(x) + g(x) = f'(x) + g'(x)
        
        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * 1 # x + b = 1 + 0 = 1
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "add")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * 1 # a + b = 0 + 1 = 1
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "add")
            b.gradient      += b_reshaped

        return
    
    def back_sub(self, prev_node, a, b, args) -> None:
        # SUB : f(x) - g(x) = f'(x) - g'(x)

        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * 1     # x - b = 1 - 0 = 1
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "sub")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * (-1)  # a - x = 0 - 1 = -1
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "sub")
            b.gradient      += b_reshaped
        
        return
    
    def back_mul(self, prev_node, a, b, args) -> None:
        # MUL : f′(x) * g(x) + f(x) * g′(x), meaning that we gotta split this formula in half for a and b.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * b_value # x * b_value = 1 * b_value = b_value
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "mul")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * a_value # a_value * x = a_value * 1 = a_value
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "mul")
            b.gradient      += b_reshaped
        
        return
    
    def back_truediv(self, prev_node, a, b, args) -> None:
        # DIV : (f'(x) * g(x) - f(x) * g'(x)) / (g(x))**2
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a_res = b_value / b_value**2    # (1 * b - a * 0) / b ** 2 = (b - 0) / b**2
        b_res = -a_value / b_value**2   # (0 * b - a * 1) / b**2 = (0 - a) / b**2

        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * a_res
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "truediv")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * b_res
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "truedivb")
            b.gradient      += b_reshaped

        return
    
    def back_floordiv(self, prev_node, a, b, args) -> None:
        # FLOORDIV : Derivatives for g(x) and f(x) is always 0.
        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * 0
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "floordiv")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * 0
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "floordiv")
            b.gradient      += b_reshaped

        return

    def back_pow(self, prev_node, a, b, args) -> None:
        # POW : for A is n * f(x)**(n-1), for B is actually (a**f(x)) * ln(a) so the constant exp rule.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a_res = b_value * (a_value ** (b_value - 1))    # b * a ** (b-1)
        b_res = (a_value ** b_value) * np.log(a_value)  # (a ** b) * ln(a)

        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * a_res
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "pow")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * b_res
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "pow")
            b.gradient      += b_reshaped

        return
    
    def back_mod(self, prev_node, a, b, args) -> None:
        # MOD : f(x) % g(x) = f'(x) - g'(x) * (f(x) // g(x))
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a_res = a_value // b_value      # 1 - 0 * (a // b)
        b_res = -(a_value // b_value)   # 0 - 1 * (a // b)

        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * a_res
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "mod")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  = prev_node.gradient * b_res
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "mod")
            b.gradient      += b_reshaped

        return
    
    def back_matmul(self, prev_node, a, b, args) -> None:
        # MATMUL : For A it's self.grad @ B.T, for B it's A.T @ self.grad (Don't ask me why, i suck at matrix operations).
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b


        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient @ b_value.T
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "matmul")
            a.gradient      += a_reshaped
        if isinstance(b, Tracer):
            b_gradient_add  =  a_value.T @ prev_node.gradient
            b_reshaped      = self.back_reshape(b_gradient_add, b.value, "matmul")
            b.gradient      += b_reshaped
        
        return

    def back_dtype(self, prev_node, a, b, args) -> None:
        # If we change dtype then the gradient is literally the same, so we just add it.

        if isinstance(a, Tracer):
            gradient = prev_node.gradient
            if prev_node.gradient.dtype != a.value.dtype:
                gradient = prev_node.gradient.astype(a.value.dtype)

            a_reshaped = self.back_reshape(gradient, a.value, "dtype")
            a.gradient += a_reshaped

        return

    def back_neg(self, prev_node, a, b, args) -> None:
        # Just like dtype, it's simple the same thing. But it's negative so -gradient
        if isinstance(a, Tracer):
            gradient = -prev_node.gradient

            a_reshaped = self.back_reshape(gradient, a.value, "neg")
            a.gradient += a_reshaped

        return

    def back_log(self, prev_node, a, b, args) -> None:
        # LOG : Should be f'(x) = 1 / x, so the gradient should be prev_node.gradient * (1 / x)

        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        if isinstance(a, Tracer):
            a_gradient_add  = prev_node.gradient * (1 / a_value)
            a_reshaped      = self.back_reshape(a_gradient_add, a.value, "log")
            a.gradient      += a_reshaped
        

        return

    def back_sum(self, prev_node, a, b, args) -> None:
        # SUM: Should always be 1.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        keepdims: bool = args["keepdims"] if "keepdims" in args else False
        grad = prev_node.gradient

        match args["axis"]:
            case None:
                a.gradient += np.ones_like(a.value) * grad
            case _:
                if not keepdims and isinstance(a, Tracer):
                    a.gradient += np.ones_like(a_value) * np.expand_dims(grad, axis = args["axis"])
                elif keepdims and isinstance(a, Tracer):
                    a.gradient += grad
        return

    def back_max(self, prev_node, a, b, args) -> None:
        # Max needs to have a mask that gets distributed by how many numbers were that max number in the array.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        keepdims: bool = args["keepdims"] if "keepdims" in args else False

        match args["axis"]:
            case None:
                mask = (a_value == np.max(a_value, keepdims = keepdims))
                mask_num = np.sum(mask, keepdims = keepdims)

                spread_gradient = prev_node.gradient / mask_num
                a.gradient += spread_gradient * mask

            case _:
                mask = (a_value == np.max(a_value, axis = args["axis"], keepdims = keepdims))
                mask_num = np.sum(mask, axis = args["axis"], keepdims = keepdims)

                prev_gradient = prev_node.gradient
                if keepdims == False:
                    prev_gradient = np.expand_dims(prev_node.gradient, axis = args["axis"])

                spread_gradient = prev_gradient / mask_num
                a.gradient += spread_gradient * mask
        return
    
    def back_mean(self, prev_node, a, b, args) -> None:
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b
        
        keepdims: bool = args["keepdims"] if "keepdims" in args else False

        match args["axis"]:
            case None:
                a.gradient += np.ones_like(a_value) * prev_node.gradient / a_value.size

            case _:
                if not keepdims and isinstance(a, Tracer):
                    a.gradient += np.ones_like(a_value) * np.expand_dims(prev_node.gradient, axis = args["axis"]) / a_value.shape["axis"]
                elif keepdims and isinstance(a, Tracer):
                    a.gradient += np.ones_like(a_value) * prev_node.gradient / a_value.shape["axis"]
        return

    def back_exp(self, prev_node, a, b, args) -> None:
        # As far as i know basically exp(parent a) (it's rare, but possible that it might be b, but i'll avoid
        # for the sake of convenience) * prev_node.gradient.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a.gradient += np.exp(a_value) * prev_node.gradient

        return

################
### Ordering ###
################

def topological_order(final_node) -> list:
    """
    Basic way to order nodes dependencies.

    :param final_node: The final node.
    """
    visited = set()
    complete = []
    def order(new_node):
        visited.add(new_node) # Checks the current one as visited so we avoid repeating the same one.
        for parent in new_node.parents_operations:
            if isinstance(parent, Tracer) and parent not in visited:
                # If not visited, we check it's parents
                order(parent)
        # For the ones who don't have parents or have already completed their cycle.
        complete.append(new_node)
        return
    order(final_node)

    return complete

