import cupyautograd as ag
import numpy as np
from time import time

class BaseLayer:
    z = None
    w = None
    b = None
    a = None
    prev_layer = None
    next_layer = None
    shape = None
    params = 0
    batch_size = None
    activation = None
    batch_size = None
    optimizer = None
    def __init__(self, w, b, shape=shape, params=0, prev_layer=None, activation='linear'):
        self.w = w
        self.b = b
        self.shape = shape
        self.params = params
        self.prev_layer = prev_layer
        if prev_layer is not None:
            prev_layer.next_layer = self
        self.activation = activation
class Layer(BaseLayer):
    def __init__(self, shape, activation, prev_layer):
        self.activation = activation
        w = ag.Var(np.random.rand(shape, prev_layer.shape) * np.sqrt(2/prev_layer.shape), require_grad=True)
        b = ag.Var(np.random.rand(shape) * np.sqrt(2/prev_layer.shape), require_grad=True)
        params = shape * prev_layer.shape
        super().__init__(w,b,shape,params,prev_layer,activation)
    def __activate(self):
        if self.activation == 'ReLU':
            self.a = ag.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + ag.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = ag.tanh(self.z)
        elif self.activation == 'softmax':
            if self.z.ndim == 1:
                self.z = ag.expand_dims(self.z, axis=0)
            mx = ag.max(self.z, axis=1)
            mx.require_grad = False
            z = (self.z.transpose(1,0)-mx).transpose(1,0)
            self.a = (ag.exp(z).transpose(1,0) / ag.sum(ag.exp(z), axis=1)).transpose(1,0)
            self.a.value = np.maximum(self.a.value, 1e-8)
        elif self.activation == 'LeakyReLU':
            self.a = ag.maximum(self.z/10, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')

        self.a.value = self.a.value.astype('float32')
        
    #compute derivative of activation function
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        self.z = (ag.dot(self.w, self.prev_layer.a.transpose(1,0))).transpose(1,0) + self.b
        self.__activate()
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def update(self):
        change_w, change_b = self.optimizer.compute_change(self.w.gradient, self.b.gradient)
        self.w.value -= change_w
        self.b.value -= change_b
        self.reset()
    def reset(self):
        self.w.reset()
        self.b.reset()

class Conv2D(BaseLayer):
    kernel_size = None
    filters = None
    padding = None
    stride = None
    def __init__(self, filters, kernel_size=3, stride=1, padding=0, activation='ReLU', prev_layer=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        p_height, p_width, p_depth = prev_layer.shape
        height = int((p_height + 2*padding - kernel_size)//stride) + 1
        width  = int((p_width  + 2*padding - kernel_size)//stride) + 1

        w = np.random.rand(filters, kernel_size, kernel_size, p_depth)/kernel_size
        b = np.random.rand(filters)/kernel_size
        w = ag.Var(w, require_grad=True)
        b = ag.Var(b, require_grad=True)

        shape = (height, width, filters)
        params = int(np.prod(w.shape) + b.shape[0])

        super().__init__(w,b,shape,params,prev_layer,activation)
    def __activate(self):
        if self.activation == 'ReLU':
            self.a = ag.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + ag.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = ag.tanh(self.z)
        elif self.activation == 'softmax':
            pass
        elif self.activation == 'LeakyReLU':
            self.a = ag.maximum(self.z/10, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')
        
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        self.z = ag.conv2d(input=self.prev_layer.a, kernel=self.w, bias=self.b, padding=self.padding, stride=self.stride)
        self.__activate()
        if self.next_layer:
            self.next_layer.feed_forward()
    def update(self):
        change_w, change_b = self.optimizer.compute_change(self.w.gradient, self.b.gradient)
        self.w.value -= change_w
        self.b.value -= change_b
        self.reset()
    def reset(self):
        self.w.reset()
        self.b.reset()
class MaxPool2D(BaseLayer):
    stride = None
    pool_size = None
    def __init__(self, pool_size=2, stride=None, prev_layer=None):
        if stride is None:
            stride = pool_size

        self.stride = stride
        self.pool_size = pool_size
        p_height, p_width, p_depth = prev_layer.shape
        depth  = p_depth
        height = int((p_height - pool_size) // stride) + 1
        width  = int((p_width  - pool_size) // stride) + 1
        shape = (height, width, depth)
        super().__init__(None,None,shape,0,prev_layer)
    
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        self.a = ag.maxpool2d(input=self.prev_layer.a, pool_size=self.pool_size, stride=self.stride)
        self.next_layer.feed_forward()
    def update(self):
        return
 
class Flatten(BaseLayer):
    def __init__(self, prev_layer):
        shape = int(np.prod(prev_layer.shape))
        super().__init__(None,None,shape,0,prev_layer)
    
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        _, height, width, depth = self.prev_layer.a.shape
        self.a = self.prev_layer.a.reshape((self.batch_size, height * width * depth))
        self.next_layer.feed_forward()
    def update(self):
        return