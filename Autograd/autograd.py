import numpy as np

class Var:
    def __init__(self, value, children=None, op=None, local_gradients=None, require_grad=False, gradient=None):
        self.value = np.array(value)
        self.shape = self.value.shape
        self.ndim = self.value.ndim
        if gradient is None:
            self.gradient = np.zeros_like(self.value).astype('float32')
        else:
            self.gradient = gradient
        self.children = children
        self.op = op
        self.local_gradients = local_gradients
        self.require_grad = require_grad
    def __getitem__(self, key):
        return Var(value=self.value[key], gradient=self.gradient[key])
    def __setitem__(self, key, value):
        if self.require_grad:
            raise("In-place operation in require-grad tensor.")
        if isinstance(value, Var):
            self.value[key] = value.value
        else:
            self.value[key] = value
    def __str__(self) -> str:
        return str(self.value)
    def __neg__(self):
        return Var(
            value=-self.value, 
            children=[self], op='neg',
            local_gradients=[lambda x: -x], 
            require_grad=self.require_grad)
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        result = self.value + other.value
        if result.ndim != other.ndim:
            n = result.ndim - other.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='+', 
                local_gradients=[lambda x: x, lambda x: np.sum(x, axis=axis)],
                require_grad=self.require_grad or other.require_grad)
        elif result.ndim != self.ndim:
            n = result.ndim - self.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='+', 
                local_gradients=[lambda x: np.sum(x, axis=axis), lambda x: x],
                require_grad=self.require_grad or other.require_grad)
        return Var(
                result,
                children=[self, other], op='+', 
                local_gradients=[lambda x: x, lambda x: x],
                require_grad=self.require_grad or other.require_grad)
    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        result = self.value * other.value
        if result.ndim > other.ndim:
            n = result.ndim - other.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='*',
                local_gradients=[lambda x: x * other.value, lambda x: np.sum(x * self.value, axis=axis)],
                require_grad=self.require_grad or other.require_grad)
        elif result.ndim > self.ndim:
            n = result.ndim - self.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='*',
                local_gradients=[lambda x: np.sum(x * other.value, axis=axis), lambda x: x * self.value],
                require_grad=self.require_grad or other.require_grad)
        return Var(
            result,
            children=[self, other], op='*',
            local_gradients=[lambda x: x * other.value, lambda x: x * self.value],
            require_grad=self.require_grad or other.require_grad)
    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        result = self.value - other.value
        if result.ndim > other.ndim:
            n = result.ndim - other.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='-',
                local_gradients=[lambda x: x, lambda x: -np.sum(x, axis=axis)],
                require_grad=self.require_grad or other.require_grad)
        elif result.ndim > self.ndim:
            n = result.ndim - self.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='-',
                local_gradients=[lambda x: np.sum(x, axis=axis), lambda x: -x],
                require_grad=self.require_grad or other.require_grad)
        return Var(
            self.value - other.value,
            children=[self, other], op='-',
            local_gradients=[lambda x: x, lambda x: -x],
            require_grad=self.require_grad or other.require_grad)
    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        result = self.value / other.value
        if result.ndim > other.ndim:
            n = result.ndim - other.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='/',
                local_gradients=[lambda x: x / other.value, lambda x: np.sum(-x * self.value/ (other.value**2), axis=axis)],
                require_grad=self.require_grad or other.require_grad)
        elif result.ndim > self.ndim:
            n = result.ndim - self.ndim
            axis = tuple(range(n))
            return Var(
                result,
                children=[self, other], op='/',
                local_gradients=[lambda x: np.sum(x / other.value, axis=axis), lambda x: -x * self.value/ (other.value**2)],
                require_grad=self.require_grad or other.require_grad)
        return Var(
            self.value / other.value,
            children=[self, other], op='/',
            local_gradients=[lambda x: x * 1/other.value, lambda x: -x * self.value / (other.value**2)],
            require_grad=self.require_grad or other.require_grad)
    def __pow__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(
            self.value ** other.value,
            children=[self, other], op='**',
            local_gradients=[lambda x: x * other.value * self.value ** (other.value - 1.0), lambda x: x * self.value ** other.value * np.log(self.value)],
            require_grad=self.require_grad or other.require_grad)
    def transpose(self, *arg):
        axis = list(arg)
        restore_axis = []
        n = self.ndim
        for i in range(n):
            j = axis.index(i)
            restore_axis.append(j)

        return Var(
            value=self.value.transpose(axis), 
            children=[self], op='transpose', 
            local_gradients=[lambda x: x.transpose(restore_axis)],
            require_grad=self.require_grad)
    def reshape(self, shape):
        return Var(
            np.reshape(self.value, shape), 
            children=[self], op='reshape', 
            local_gradients=[lambda x: x.reshape(self.shape)], 
            require_grad=self.require_grad)
    def backward(self, grad=1.0):
        if not self.require_grad:
            return
        grad = np.array(grad).astype('float32')
        if grad.shape != self.shape:
            if self.ndim == 0:
                grad = np.sum(grad)
            elif self.ndim > grad.ndim:
                grad = np.broadcast_to(grad, self.value.shape)
            else:
                grad = np.sum(grad, axis=tuple(np.arange(self.ndim)))

        self.gradient += grad
        if self.children is not None:
            for child, local_grad in zip(self.children, self.local_gradients):
                child.backward(grad=local_grad(grad))
    def reset(self):
        self.gradient = np.zeros_like(self.value)
def dot(Var1, Var2):
    Var1 = Var1 if isinstance(Var1, Var) else Var(Var1)
    Var2 = Var2 if isinstance(Var2, Var) else Var(Var2)
    try:
        v1, v2 = Var1.value, Var2.value
        lambda1, lambda2 = None, None
        if v1.ndim * v2.ndim == 0:
            return Var1 * Var2
        
        elif v2.ndim == 1:
            n = v1.ndim
            lambda1 = lambda x: np.repeat(np.expand_dims(x, axis=n-1), v2.shape[0], axis=n-1) * v2
            new_order = list(np.arange(n-1))
            new_order.insert(0, n-1)
            tmp = v1.transpose(new_order)
            axis = tuple(range(1,n))
            lambda2 = lambda x: np.sum(tmp * x, axis=axis)

        elif Var1.ndim == Var2.ndim:
            lambda1 = lambda x: np.dot(x, v2.T)
            lambda2 = lambda x: np.dot(v1.T, x)

        return Var(np.dot(v1, v2), 
                   children=[Var1, Var2], op='dot', 
                   local_gradients=[lambda1, lambda2], 
                   require_grad=Var1.require_grad or Var2.require_grad)
        
    except NameError:
        print("incompatible dimension")
def outer(Var1, Var2):
    Var1 = Var1 if isinstance(Var1, Var) else Var(Var1)
    Var2 = Var2 if isinstance(Var2, Var) else Var(Var2)
    return Var(
        np.outer(Var1.value, Var2.value),
        children=[Var1, Var2], op='outer',
        local_gradients=[lambda x: np.dot(x, Var2.value), lambda x: np.dot(x.T, Var1.value)],
        require_grad=Var1.require_grad or Var2.require_grad
    )

def abs(a: Var):
    return Var(np.abs(a.value), children=[a], op='abs', local_gradients=[lambda x: x * np.sign(a.value)], require_grad=a.require_grad)
def argmax(a: Var, axis):
    return (np.argmax(a.value, axis=axis))
def expand_dims(a: Var, axis):
    new_value = np.expand_dims(a.value, axis)
    return Var(new_value, 
               children=[a], op='expand dims', 
               local_gradients=[lambda x: x.reshape(a.shape)], 
               require_grad=a.require_grad)

def exp(A):
    A = A if isinstance(A, Var) else Var(A)
    return Var(np.exp(A.value), children=[A], op='exp', local_gradients=[lambda x: x * np.exp(A.value)], require_grad=A.require_grad)
def log(A):
    A = A if isinstance(A, Var) else Var(A)
    return Var(np.log(A.value), children=[A], op='log', local_gradients=[lambda x: x / A.value], require_grad=A.require_grad)
def sum(A, axis=None):
    A = A if isinstance(A, Var) else Var(A)
    if axis is None:
        return Var(np.sum(A.value), children=[A], op='sum', local_gradients=[lambda x: x * np.ones(shape=A.shape)], require_grad=A.require_grad)
    
    if isinstance(axis, int):
        axis = [axis]
    def help(value, axes, shape):
        for axis in axes:
            n = value.ndim
            value = np.expand_dims(value, axis=n)
            value = np.repeat(value, shape[axis], axis=n)
        return value
    x = list(set(range(A.ndim)).difference(set(axis)))
    new_axis = x + axis
    reverse_axis = []
    for i in range(A.ndim):
        reverse_axis.append(new_axis.index(i))
    return Var(np.sum(A.value, axis=tuple(axis)), children=[A], local_gradients=[lambda x: help(x, axis, A.shape).transpose(reverse_axis)], op='sum', require_grad=A.require_grad)
def maximum(A, B):
    A = A if isinstance(A, Var) else Var(A)
    B = B if isinstance(B, Var) else Var(B)
    return Var(np.maximum(A.value, B.value), children=[A, B],
                local_gradients=[lambda x: x * np.where(A.value >= B.value, 1, 0), lambda x: x * np.where(A.value < B.value, 1, 0)],
                require_grad=A.require_grad)
def max(A, axis=None):
    A = A if isinstance(A, Var) else Var(A)
    return Var(np.max(A.value, axis=axis), children=A,
               local_gradients=[lambda x: np.where(A.value == np.max(A.value, axis=axis), 1, 0) * x],
               require_grad=A.require_grad)
def tanh(A):
    A = A if isinstance(A, Var) else Var(A)
    return (- exp(-A*2) + 1)/(exp(-A*2) + 1)
def conv2d(input: Var, kernel: Var, bias:Var, padding, stride):
    k = kernel.value
    b = bias.value
    filters, ksize, _, _ = k.shape
    batch_size, i_height, i_width, _ = input.shape
    padding_input = np.pad(input.value, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    o_height = (i_height + 2 * padding - ksize)//stride + 1
    o_width  = (i_width  + 2 * padding - ksize)//stride + 1
    tmp_padding = np.expand_dims(padding_input, axis=1)
    tmp_padding = np.repeat(tmp_padding, filters, axis=1)
    output = np.zeros(shape=(batch_size, o_height, o_width, filters))
    for i in range(0, i_height - ksize + 1, stride):
        for j in range(0, i_width - ksize + 1, stride):
            tmp = tmp_padding[:,:,i:i+ksize,j:j+ksize,:]
            ii = i//stride
            jj = j//stride
            output[:,ii,jj,:] = np.sum((tmp * k), axis=(2,3,4)) + b   
    
    return Var(output, children=[input, kernel, bias], op='conv2d', 
               local_gradients=[lambda x: conv2d_input_backprop(x, padding_input, k, stride, padding), 
                                lambda x: conv2d_kernel_backprop(x, padding_input, k, stride),
                                lambda x: np.sum(x, axis=(0,1,2))], 
                require_grad=input.require_grad or kernel.require_grad or bias.require_grad)

def conv2d_input_backprop(gradient, padding_a, kernel, stride, padding):
    a = np.zeros_like(padding_a)
    batch_size,m,n,d = padding_a.shape
    tmp_kernel = np.expand_dims(kernel, axis=0)
    tmp_kernel = np.repeat(tmp_kernel, batch_size, axis=0)
    kernel_size = kernel.shape[1]
    for i in range(0, m-kernel_size, stride):
        for j in range(0, n-kernel_size, stride):
            ii = i//stride
            jj = j//stride
            tmp = (tmp_kernel.transpose(2,3,4,0,1) * gradient[:,ii,jj,:]).transpose(3,4,0,1,2)
            tmp = np.sum(tmp, axis=1)
            a[:,i:i+kernel_size,j:j+kernel_size,:] += tmp

    return a[:,padding:m-padding,padding:n-padding,:]            

def conv2d_kernel_backprop(gradient, padding_a, kernel, stride):
    d_w = np.zeros_like(kernel)
    kernel_size = kernel.shape[1]
    filters = kernel.shape[0]
    _,m,n,_ = padding_a.shape
    tmp_padding_a = np.expand_dims(padding_a, axis=4)
    tmp_padding_a = np.repeat(tmp_padding_a, filters, axis=4)
    for i in range(0, m-kernel_size, stride):
        for j in range(0, n-kernel_size, stride):
            tmp = tmp_padding_a[:,i:i+kernel_size,j:j+kernel_size,:,:]
            ii  = i//stride
            jj  = i//stride
            tmp = (tmp.transpose(1,2,3,0,4) * gradient[:,ii,jj,:]).transpose(3,4,0,1,2)
            d_w += np.sum(tmp, axis=0)
    
    return d_w
          
def maxpool2d(input: Var, pool_size, stride=None, padding=0):
    batch_size, i_height, i_width, depth = input.shape
    o_height = (i_height + 2 * padding - pool_size)//stride + 1
    o_width  = (i_width  + 2 * padding - pool_size)//stride + 1
    out = np.zeros(shape=(batch_size, o_height, o_width, depth))
    for i in range(0, o_height):
        for j in range(0, o_width):
            out[:,i,j,:] = np.max(input.value[:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size,:], axis=(1,2))

    return Var(out, children=[input], op='maxpool2d',
               local_gradients=[lambda x: maxpool2d_backprop(x, input.value, out.shape, pool_size, stride)],
               require_grad=input.require_grad)
def maxpool2d_backprop(gradient, input: np.ndarray, output_shape, pool_size, stride):
    local_gradient = np.zeros_like(input)
    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            a = input[:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size,:]
            b, h, w, d = a.shape
            tmp = a.reshape(b, h*w, d)
            mx  = np.max(tmp, axis=1, keepdims=True)
            mx  = np.repeat(mx, h*w, axis=1)
            mx  = mx.reshape(b, h, w, d)
            max_position = np.where(a == mx, 1, 0)
            local_gradient[:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size,:] = max_position
    
    return (local_gradient.transpose(1,2,0,3) * gradient).transpose(2,0,1,3)

def main():
    a = Var([[1.0, 2.0], [3.0, 4.5]], require_grad=True)
    b = Var([2, 1.75], require_grad=True)
    c = a * b
    c.backward()
    print(b.gradient)
if __name__ == '__main__':
    main()