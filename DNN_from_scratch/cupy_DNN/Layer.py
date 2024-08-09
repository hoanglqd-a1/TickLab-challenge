import numpy as np
import cupy as cp
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
        w = cp.random.uniform(-1, 1, size=(shape, prev_layer.shape)) * cp.sqrt(6/(prev_layer.shape + shape))
        b = cp.random.uniform(-1, 1, size=shape) * cp.sqrt(6/(prev_layer.shape + shape))
        params = shape * prev_layer.shape
        super().__init__(w,b,shape,params,prev_layer,activation)
    def __activate(self):
        if self.activation == 'ReLU':
            self.a = cp.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + cp.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = cp.tanh(self.z)
        elif self.activation == 'softmax':
            if self.z.ndim == 1:
                self.z = cp.expand_dims(self.z, axis=0)
            z = (self.z.T - cp.max(self.z, axis=1)).T
            self.a = (cp.exp(z).T / cp.sum(cp.exp(z), axis=1)).T
            self.a = cp.maximum(self.a, 1e-8)
        elif self.activation == 'LeakyReLU':
            self.a = cp.maximum(self.z/100, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')
        
        self.a = cp.asnumpy(self.a.astype('float64'))
    #compute derivative of activation function
    def __gradient_of_activation(self):
        d_a = None
        if self.activation == 'ReLU':
            d_a = np.where(self.a > 0, 1, 0)
        elif self.activation == 'sigmoid':
            d_a = self.a * (1 - self.a)
        elif self.activation == 'tanh':
            d_a = 1 - self.a**2
        elif self.activation == 'softmax':
            d_a = np.zeros(shape=([self.a.shape[0], self.a.shape[1], self.a.shape[1]]))
            for i in range(self.a.shape[0]):
                d_a[i] = np.diagflat(self.a[i]) - np.outer(self.a[i], self.a[i])
        elif self.activation == 'LeakyReLU':
            d_a = np.where(self.a > 0, 1, 0.01)
        elif self.activation == 'linear':
            d_a = np.ones(shape=self.shape)
        else:
            raise ValueError('Invalid activation function')

        return d_a
    def info(self):
        return f"Fully connected layer - Output shape: {self.shape} - Parameters: {self.params}"
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        p_a = cp.asarray(self.prev_layer.a)
        self.z = cp.dot(self.w, p_a.T).T + self.b
        self.__activate()
        print('fully', self.a[:5,:15])
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def back_propagation(self, e):
        gradient = self.__gradient_of_activation()
        if self.activation == 'softmax':
            pass
        else:
            e = e * gradient
        
        e = cp.asarray(e)

        p_a = cp.asarray(self.prev_layer.a)
        d_w = cp.dot(e.T, p_a)
        d_b = cp.sum(e, axis=0)
        
        e = cp.dot(self.w.T, e.T).T
        e = cp.asnumpy(e)

        change_w, change_b = self.optimizer.compute_change(d_w, d_b)

        self.w -= change_w / self.batch_size
        self.b -= change_b / self.batch_size

        #previous layer call back propagation
        self.prev_layer.back_propagation(e)

class Conv2D(BaseLayer):
    kernel_size = None
    filters = None
    padding = None
    stride = None
    def __init__(self, filters, kernel_size=3, stride=1, padding=0, activation='linear', prev_layer=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        p_height, p_width, p_depth = prev_layer.shape
        height = int((p_height + 2*padding - kernel_size)//stride) + 1
        width  = int((p_width  + 2*padding - kernel_size)//stride) + 1

        w = cp.random.uniform(-1, 1, size=(filters, kernel_size, kernel_size, p_depth))*cp.sqrt(6/(p_depth*kernel_size**2+1))
        b = cp.random.uniform(-1, 1, size=filters)*cp.sqrt(6/(p_depth*kernel_size**2+1))

        shape = (height, width, filters)
        params = int(np.prod(w.shape) + b.shape[0])

        super().__init__(w,b,shape,params,prev_layer,activation)
    def __activate(self):
        if self.activation == 'ReLU':
            self.a = cp.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + cp.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = cp.tanh(self.z)
        elif self.activation == 'LeakyReLU':
            self.a = cp.maximum(self.z/100, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')
        
        self.a = cp.asnumpy(self.a).astype('float64')
    
    def __gradient_of_activation(self):
        d_a = None
        if self.activation == 'ReLU':
            d_a = np.where(self.a > 0, 1, 0)
        elif self.activation == 'sigmoid':
            d_a = self.a * (1 - self.a)
        elif self.activation == 'tanh':
            d_a = 1 - self.a**2
        elif self.activation == 'LeakyReLU':
            d_a = np.where(self.a > 0, 1, 0.01)
        elif self.activation == 'linear':
            d_a = np.ones(shape=self.a.shape)
        else:
            raise ValueError('Invalid activation function')

        return d_a
    def info(self):
        return f"Conv2D - Output shape: {self.shape} - Parameters: {self.params}"
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        kernel = self.kernel_size
        height, width, depth = self.shape
        self.z = cp.zeros(shape=[self.batch_size, height, width, depth])

        self.padding_a = np.pad(self.prev_layer.a, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        _,m,n,_ = self.padding_a.shape
        for i in range(0, m - kernel + 1, self.stride):
            for j in range(0, n - kernel + 1, self.stride):
                tmp_padding_a = self.padding_a[:,i:i+kernel,j:j+kernel,:]
                tmp = np.expand_dims(tmp_padding_a, axis=1)
                tmp = cp.asarray(np.repeat(tmp, self.filters, axis=1))
                ii = i//self.stride
                jj = j//self.stride
                self.z[:,ii,jj,:] = cp.sum((tmp * self.w), axis=(2,3,4)) + self.b

        self.__activate()
        print('conv2d', self.a[0, :5, :5, 0])
        if self.next_layer: 
            self.next_layer.feed_forward()

    def back_propagation(self, e):
        gradient = self.__gradient_of_activation()
        e = e * gradient
        
        e = cp.asarray(e)
        kernel = self.kernel_size
        _,m,n,_ = self.padding_a.shape
        d_w = cp.zeros(shape=self.w.shape)
        d_b = cp.zeros(shape=self.b.shape)
        for i in range(0, m - kernel + 1, self.stride):
            for j in range(0, n - kernel + 1, self.stride):
                tmp = self.padding_a[:,i:i+kernel,j:j+kernel,:]
                tmp = cp.asarray(np.repeat(np.expand_dims(tmp, axis=4), self.filters, axis=4))
                ii  = i//self.stride
                jj  = j//self.stride
                tmp = (tmp.transpose(1,2,3,0,4) * e[:,ii,jj,:]).transpose(3,4,0,1,2)
                tmp = cp.sum(tmp, axis=0)
                d_w += tmp
        d_b = cp.sum(e, axis=(0,1,2))
        
        a = cp.zeros(shape=self.padding_a.shape)
        tmp_kernel = cp.expand_dims(self.w, axis=0)
        tmp_kernel = cp.repeat(tmp_kernel, self.batch_size, axis=0)
        for i in range(0, m - kernel + 1, self.stride):
            for j in range(0, n - kernel + 1, self.stride):
                ii  = i//self.stride
                jj  = j//self.stride
                tmp = (tmp_kernel.transpose(2,3,4,0,1) * e[:,ii,jj,:]).transpose(3,4,0,1,2)
                tmp = cp.sum(tmp, axis=1)
                a[:,i:i+kernel,j:j+kernel,:] += tmp
        
        p = self.padding
        p_height, p_width, _ = self.prev_layer.shape
        e = a[:,p:p_height+p,p:p_width+p,:]

        change_w, change_b = self.optimizer.compute_change(d_w, d_b)

        self.w -= change_w/self.batch_size
        self.b -= change_b/self.batch_size

        e = cp.asnumpy(e)

        self.prev_layer.back_propagation(e)

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
        height = (p_height - pool_size)//stride + 1
        width  = (p_width  - pool_size)//stride + 1
        shape = (height, width, depth)
        super().__init__(None,None,shape,0,prev_layer)
    
    def info(self):
        return f"MaxPool2D - Output shape: {self.shape}"
    
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size

        (height, width, depth) = self.shape
        self.a = np.zeros(shape=(self.batch_size, height, width, depth))
        s = self.stride
        p = self.pool_size
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                self.a[:,i,j,:] = np.max(self.prev_layer.a[:,i*s:i*s+p,j*s:j*s+p,:], axis=(1,2))

        self.next_layer.feed_forward()

    def back_propagation(self, e):
        p_height, p_width, p_depth = self.prev_layer.shape
        d_A = np.zeros(shape=(self.batch_size, p_height, p_width, p_depth))
        s = self.stride
        p = self.pool_size
        for i in range(0, self.shape[0]):       
            for j in range(0, self.shape[1]):
                a = self.prev_layer.a[:,i*s:i*s+p,j*s:j*s+p,:]
                b, h, w, d = a.shape
                tmp = a.reshape(b, h*w, d)
                mx  = np.max(tmp, axis=1, keepdims=True)
                mx  = np.repeat(mx, h*w, axis=1)
                mx  = mx.reshape(b, h, w, d) 
                max_position = np.where(a==mx, 1, 0)
                d_A[:,i*s:i*s+p,j*s:j*s+p,:] += (max_position.transpose(1,2,0,3)*e[:,i,j,:]).transpose(2,0,1,3)
        e = d_A

        self.prev_layer.back_propagation(e)
    
class Flatten(BaseLayer):
    def __init__(self, prev_layer):
        shape = int(np.prod(prev_layer.shape))
        super().__init__(None,None,shape,0,prev_layer)
    def info(self):
        return f"Flatten - Output shape: {self.shape}"
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        self.a = self.prev_layer.a.reshape(self.batch_size, -1)
        self.next_layer.feed_forward()
    
    def back_propagation(self, e):
        e = e.reshape(self.prev_layer.a.shape)
        self.prev_layer.back_propagation(e)