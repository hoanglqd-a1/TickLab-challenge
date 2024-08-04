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
        w = np.random.rand(shape, prev_layer.shape) * np.sqrt(1/prev_layer.shape)
        b = np.random.rand(shape) * np.sqrt(1/prev_layer.shape)
        params = shape * prev_layer.shape
        super().__init__(w,b,shape,params,prev_layer,activation)
    def __activate(self):
        if self.activation == 'ReLU':
            self.a = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + np.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = np.tanh(self.z)
        elif self.activation == 'softmax':
            if self.z.ndim == 1:
                self.z = np.array([self.z])
            z = (self.z.T - np.max(self.z, axis=1)).T
            self.a = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
            self.a = np.maximum(self.a, 1e-8)
        elif self.activation == 'LeakyReLU':
            self.a = np.maximum(self.z/10, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')
        
        self.a = self.a.astype('float32')
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
            d_a = np.where(self.a > 0, 1, 0.1)
        elif self.activation == 'linear':
            d_a = np.ones(shape=self.shape)
        else:
            raise ValueError('Invalid activation function')

        return d_a
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        p_a = self.prev_layer.a
        self.z = np.dot(self.w, p_a.T).T + self.b
        self.__activate()
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def back_propagation(self, e):
        gradient = self.__gradient_of_activation()
        if self.activation == 'softmax':
            pass
        else:
            e = e * gradient
        
        d_w = np.zeros(shape=self.w.shape)
        d_b = np.zeros(shape=self.b.shape)

        p_a = self.prev_layer.a
        for i in range(e.shape[0]):
            d_w += np.outer(e[i], p_a[i])
        d_b = np.sum(e, axis=0)
        
        e = np.dot(self.w.T, e.T).T
        change_w, change_b = self.optimizer.compute_change(d_w, d_b)

        self.w -= change_w / self.batch_size
        self.b -= change_b / self.batch_size

        #previous layer call back propagation
        self.prev_layer.back_propagation(e)

class Conv2D(BaseLayer):
    change_w = 0.0
    change_b = 0.0

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
        height = int((p_height + 2 * padding - kernel_size) // stride) + 1
        width  = int((p_width  + 2 * padding - kernel_size) // stride) + 1

        w = np.random.rand(filters, kernel_size, kernel_size, p_depth)/kernel_size
        b = np.random.rand(filters)/kernel_size

        shape = (height, width, filters)
        params = int(np.prod(w.shape) + b.shape[0])

        super().__init__(w,b,shape,params,prev_layer,activation)
    def __activate(self):
        if self.activation == 'ReLU':
            self.a = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + np.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = np.tanh(self.z)
        elif self.activation == 'softmax':
            self.a = self.z
        elif self.activation == 'LeakyReLU':
            self.a = np.maximum(self.z/10, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')
        
        self.a = self.a.astype('float64')
    
    def __gradient_of_activation(self):
        d_a = None
        if self.activation == 'ReLU':
            d_a = np.where(self.z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            d_a = self.a * (1 - self.a)
        elif self.activation == 'tanh':
            d_a = 1 - self.a**2
        elif self.activation == 'softmax':
            pass
        elif self.activation == 'LeakyReLU':
            d_a = np.where(self.z > 0, 1, 0.1)
        elif self.activation == 'linear':
            d_a = np.ones(shape=self.a.shape)
        else:
            raise ValueError('Invalid activation function')

        return d_a
    
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        height, width, depth = self.shape
        self.z = np.zeros(shape=[self.batch_size, height, width, depth])

        self.padding_a = np.pad(self.prev_layer.a, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        _,m,n,_ = self.padding_a.shape
        kernel = self.kernel_size

        tmp_padding_a = np.expand_dims(self.padding_a, axis=1)
        tmp_padding_a = np.repeat(tmp_padding_a, self.filters, axis=1)
        w = self.w
        b = self.b
        for i in range(0, m - kernel + 1, self.stride):
            for j in range(0, n - kernel + 1, self.stride):
                tmp = tmp_padding_a[:,:,i:i+kernel,j:j+kernel,:]
                ii = i//self.stride
                jj = j//self.stride
                self.z[:,ii,jj,:] = np.sum((tmp * w), axis=(2,3,4)) + b

        self.__activate()

        if self.next_layer:
            self.next_layer.feed_forward()

    def back_propagation(self, e):
        gradient = self.__gradient_of_activation()
        if self.activation == 'softmax':
            pass
        else:
            e = e * gradient
        
        kernel = self.kernel_size
        _,m,n,_ = self.padding_a.shape
        padding_a = self.padding_a
        d_w = np.zeros(shape=self.w.shape)
        d_b = np.zeros(shape=self.b.shape)
        tmp_padding_a = np.expand_dims(padding_a, axis=4)
        tmp_padding_a = np.repeat(tmp_padding_a, self.filters, axis=4)
        for i in range(0, m - kernel + 1, self.stride):
            for j in range(0, n - kernel + 1, self.stride):
                tmp = tmp_padding_a[:,i:i+kernel,j:j+kernel,:,:]
                ii  = i//self.stride
                jj  = j//self.stride
                tmp = (tmp.transpose(1,2,3,0,4) * e[:,ii,jj,:]).transpose(3,4,0,1,2)
                tmp = np.sum(tmp, axis=0)
                d_w += tmp
        d_b = np.sum(e, axis=(0,1,2))
        
        a = np.zeros(shape=self.padding_a.shape)
        tmp_kernel = np.expand_dims(self.w, axis=0)
        tmp_kernel = np.repeat(tmp_kernel, self.batch_size, axis=0)
        for i in range(0, m - kernel + 1, self.stride):
            for j in range(0, n - kernel + 1, self.stride):
                ii  = i//self.stride
                jj  = j//self.stride
                tmp = (tmp_kernel.transpose(2,3,4,0,1) * e[:,ii,jj,:]).transpose(3,4,0,1,2)
                tmp = np.sum(tmp, axis=1)
                a[:,i:i+kernel,j:j+kernel,:] += tmp
        
        p = self.padding
        p_height, p_width, _ = self.prev_layer.shape
        e = a[:,p:p_height+p,p:p_width+p,:]

        change_w, change_b = self.optimizer.compute_change(d_w, d_b)

        self.w -= change_w / self.batch_size
        self.b -= change_b / self.batch_size

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
        height = int((p_height - pool_size) // stride) + 1
        width  = int((p_width  - pool_size) // stride) + 1
        shape = (height, width, depth)
        super().__init__(None,None,shape,0,prev_layer)
    
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
                d_A[:,i*s:i*s+p,j*s:j*s+p,:] += (max_position.transpose(1,2,0,3) * e[:,i,j,:]).transpose(2,0,1,3)
        e = d_A

        self.prev_layer.back_propagation(e)
    
class Flatten(BaseLayer):
    def __init__(self, prev_layer):
        shape = int(np.prod(prev_layer.shape))
        super().__init__(None,None,shape,0,prev_layer)
    
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        self.a = self.prev_layer.a.reshape(self.batch_size, -1)
        self.next_layer.feed_forward()
    
    def back_propagation(self, e):
        e = e.reshape(self.prev_layer.a.shape)
        self.prev_layer.back_propagation(e)