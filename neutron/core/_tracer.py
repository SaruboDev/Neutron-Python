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
        return f"{type(self.value)}[{np.shape(self.value)}]"
    
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
    def backwards(self) -> None:
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
        for tracer in order:
            if len(tracer.parents_operations) > 0:
                if tracer.parents_operations[1] is not None:
                    a = tracer.parents_operations[0]
                    b = tracer.parents_operations[2]
                    tracer.parents_operations[1](tracer, a, b)

    def back_add(self, prev_node, a, b) -> None:
        # ADD : f(x) + g(x) = f'(x) + g'(x)
        
        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * 1 # x + b = 1 + 0 = 1
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * 1 # a + b = 0 + 1 = 1

        return
    
    def back_sub(self, prev_node, a, b) -> None:
        # SUB : f(x) - g(x) = f'(x) - g'(x)

        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * 1     # x - b = 1 - 0 = 1
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * (-1)  # a - x = 0 - 1 = -1
        
        return
    
    def back_mul(self, prev_node, a, b) -> None:
        # MUL : f′(x) * g(x) + f(x) * g′(x), meaning that we gotta split this formula in half for a and b.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * b_value # x * b_value = 1 * b_value = b_value
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * a_value # a_value * x = a_value * 1 = a_value
        
        return
    
    def back_truediv(self, prev_node, a, b) -> None:
        # DIV : (f'(x) * g(x) - f(x) * g'(x)) / (g(x))**2
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a_res = b_value / b_value**2    # (1 * b - a * 0) / b ** 2 = (b - 0) / b**2
        b_res = -a_value / b_value**2   # (0 * b - a * 1) / b**2 = (0 - a) / b**2

        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * a_res
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * b_res

        return
    
    def back_floordiv(self, prev_node, a, b) -> None:
        # FLOORDIV : Derivatives for g(x) and f(x) is always 0.
        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * 0
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * 0

        return

    def back_pow(self, prev_node, a, b) -> None:
        # POW : for A is n * f(x)**(n-1), for B is actually (a**f(x)) * ln(a) so the constant exp rule.
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a_res = b_value * (a_value ** (b_value - 1))    # b * a ** (b-1)
        b_res = (a_value ** b_value) * np.log(a_value)  # (a ** b) * ln(a)

        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * a_res
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * b_res

        return
    
    def back_mod(self, prev_node, a, b) -> None:
        # MOD : f(x) % g(x) = f'(x) - g'(x) * (f(x) // g(x))
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b

        a_res = a_value // b_value      # 1 - 0 * (a // b)
        b_res = -(a_value // b_value)   # 0 - 1 * (a // b)

        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient * a_res
        if isinstance(b, Tracer):
            b.gradient += prev_node.gradient * b_res

        return
    
    def back_matmul(self, prev_node, a, b) -> None:
        # MATMUL : For A it's self.grad @ B.T, for B it's A.T @ self.grad (Don't ask me why, i suck at matrix operations).
        a_value = a.value if isinstance(a, Tracer) else a
        b_value = b.value if isinstance(b, Tracer) else b


        if isinstance(a, Tracer):
            a.gradient += prev_node.gradient @ b_value.T
        if isinstance(b, Tracer):
            b.gradient += a_value.T @ prev_node.gradient
        
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

