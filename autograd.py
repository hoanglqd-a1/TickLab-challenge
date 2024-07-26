import numpy as np

class Var:
    def __init__(self, value, gradient=None, children=None, op=None, local_gradients=None, require_grad=False):
        self.value = np.array(value)
        self.shape = self.value.shape
        self.ndim = self.value.ndim
        self.gradient = gradient or np.zeros_like(self.value).astype('float32')
        self.children = children
        self.op = op
        self.local_gradients = local_gradients
        self.require_grad = require_grad
    def __getitem__(self, key):
        return Var(value=self.value[key], gradient=self.gradient[key])
    def __setitem__(self, key, value):
        self[key] = value
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(
            self.value + other.value,
            children=[self, other], op='+', 
            local_gradients=[np.ones(shape=self.shape), np.ones(shape=other.shape)],
            require_grad=self.require_grad or other.require_grad)

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(
            self.value * other.value,
            children=[self, other], op='*',
            local_gradients=[other.value, self.value],
            require_grad=self.require_grad or other.require_grad)

    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(
            self.value - other.value,
            children=[self, other], op='-',
            local_gradients=[np.ones(shape=self.shape), -np.ones(shape=other.shape)],
            require_grad=self.require_grad or other.require_grad)

    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(
            self.value / other.value,
            children=[self, other], op='/',
            local_gradients=[1/other.value, -self.value / other.value**2],
            require_grad=self.require_grad or other.require_grad)
    
    def __pow__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(
            self.value ** other.value,
            children=[self, other], op='**',
            local_gradients=[other.value * self.value ** (other.value - 1), self.value ** other.value * np.log(self.value)],
            require_grad=self.require_grad or other.require_grad)
    def dot(self, other):
        other = other if isinstance(other, Var) else Var(other)
        value = np.dot(self.value, other.value)
        return Var(
            np.dot(self.value, other.value), op='dot',
            local_gradients=[np.outer(self.value, other.value), np.dot(value, other.value.T).T],
            require_grad=self.require_grad or other.require_grad
        )
    def outer(self, other):
        pass
    
    def backward(self, grad=1.0):
        if not self.require_grad:
            return
        self.gradient += grad
        if self.children is not None:
            for child, local_grad in zip(self.children, self.local_gradients):
                child.backward(grad=local_grad * grad)

a = Var([1,2], require_grad=True)
b = Var([[1,2],[3,4]], require_grad=True)
c = a * b
c.backward()