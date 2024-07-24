import numpy as np

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
        prev_layer.next_layer = self
        self.activation = activation
class Layer(BaseLayer):
    def __init__(self, shape, activation, prev_layer):
        self.activation = activation
        w = np.random.rand(shape, prev_layer.shape) * np.sqrt(2/prev_layer.shape)
        b = np.random.rand(shape) * np.sqrt(2/prev_layer.shape)
        params = shape * prev_layer.shape
        super().__init__(w,b,shape,params,prev_layer,activation)
    def activate(self):
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
            self.a = np.maximum(self.a, 1e-6)
        elif self.activation == 'LeakyReLU':
            self.a = np.maximum(self.z/10, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')
        
        self.a = self.a.astype('float64')

    #compute derivative of activation function
    def derivative_of_activation(self):
        d_a = None
        if self.activation == 'ReLU':
            d_a = np.where(self.z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            d_a = self.a * (1 - self.a)
        elif self.activation == 'tanh':
            d_a = 1 - self.a**2
        elif self.activation == 'softmax':
            d_a = np.zeros(shape=([self.a.shape[0], self.a.shape[1], self.a.shape[1]]))
            for i in range(self.a.shape[0]):
                d_a[i] = np.diagflat(self.a[i]) - np.outer(self.a[i], self.a[i])
        elif self.activation == 'LeakyReLU':
            d_a = np.where(self.z > 0, 1, 0.1)
        elif self.activation == 'linear':
            d_a = np.ones(shape=self.shape)
        else:
            raise ValueError('Invalid activation function')

        return d_a
    def feed_forward(self):
        self.batch_size = self.prev_layer.batch_size
        self.z = np.dot(self.w, self.prev_layer.a.T).T + self.b
        self.activate()
        #print('layer:', np.mean(self.a[0]))
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def back_propagation(self, e):
        MOMENTUM = 0.9
        d_a = self.derivative_of_activation()
        if self.activation == 'softmax':
            pass
        else:
            e = e * d_a
        
        d_w = np.zeros(shape=self.w.shape)
        d_b = np.zeros(shape=self.b.shape)

        for i in range(e.shape[0]):
            d_w += np.outer(e[i], self.prev_layer.a[i])
        d_b = np.sum(e, axis=0)
        
        e = np.dot(self.w.T, e.T).T

        change_w, change_b = self.optimizer.compute_change(d_w, d_b)

        self.w -= change_w / self.batch_size
        self.b -= change_b / self.batch_size

        #previous layer call back propagation
        self.prev_layer.back_propagation(e)

class Conv2D(BaseLayer):
    z = None
    a = None

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
        b = np.random.rand(filters)

        shape = (height, width, filters)
        params = int(np.prod(w.shape) + b.shape[0])

        super().__init__(w,b,shape,params,prev_layer,activation)
    def activate(self):
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
    
    def derivative_of_activation(self):
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

        """for filter in range(self.filters):
            for i in range(0, m - self.kernel_size + 1, self.stride):
                for j in range(0, n - self.kernel_size + 1, self.stride):
                    ii = i//self.stride
                    jj = j//self.stride
                    self.z[:,ii,jj,filter] = np.sum((self.padding_a[:,i:i+kernel,j:j+kernel,:] * self.w[filter]), axis=(1,2,3)) + self.b[filter]"""
        
        #this following code does the same behaviour as the above one, but faster

        for i in range(0, m - self.kernel_size + 1, self.stride):
            for j in range(0, n - self.kernel_size + 1, self.stride):
                tmp = np.expand_dims(self.padding_a[:,i:i+kernel,j:j+kernel,:], axis=1)
                tmp = np.repeat(tmp, self.filters, axis=1)
                ii = i//self.stride
                jj = j//self.stride
                self.z[:,ii,jj,:] = np.sum((tmp * self.w), axis=(2,3,4)) + self.b

        self.activate()
        #print('conv:', np.mean(self.a[0][0]))
        if self.next_layer:
            self.next_layer.feed_forward()

    def back_propagation(self, e):
        d_a = self.derivative_of_activation()
        if self.activation == 'softmax':
            pass
        else:
            e = e * d_a
        
        kernel = self.kernel_size
        _,n,m,_ = self.padding_a.shape
        padding_a = self.padding_a
        d_w = np.zeros(shape=self.w.shape)
        d_b = np.zeros(shape=self.b.shape)
 
        """for filter in range(self.filters):
            for i in range(0, m - self.kernel_size + 1, self.stride):
                for j in range(0, n - self.kernel_size + 1, self.stride):
                    tmp = (padding_a[:,i:i+kernel,j:j+kernel,:].transpose(1,2,3,0) * e[:,i,j,filter].reshape(-1)).transpose(3,0,1,2)
                    tmp = np.sum(tmp, axis=0)

                    d_w[filter] += tmp"""
        
        for i in range(0, m - self.kernel_size + 1, self.stride):
            for j in range(0, n - self.kernel_size + 1, self.stride):
                tmp = np.expand_dims(padding_a[:,i:i+kernel,j:j+kernel,:], axis=4)
                tmp = np.repeat(tmp, self.filters, axis=4)
                ii = i//self.stride
                jj = j//self.stride
                tmp = (tmp.transpose(1,2,3,0,4) * e[:,ii,jj,:]).transpose(3,4,0,1,2)
                tmp = np.sum(tmp, axis=0)

                d_w += tmp
        
        d_b = np.sum(e, axis=(0,1,2))/(e.shape[1] * e.shape[2])
        a = np.zeros(shape=self.padding_a.shape)

        """for filter in range(self.filters):
            for i in range(0, m - self.kernel_size + 1, self.stride):
                for j in range(0, n - self.kernel_size + 1, self.stride):
                    tmp = np.repeat(self.w[filter].reshape(1, x, y, z), self.batch_size, axis=0)
                    tmp = (tmp.transpose(1,2,3,0) * e[:,i,j,filter]).transpose(3,0,1,2)
                    a[:,i:i+kernel,j:j+kernel,:] += tmp"""

        #the following code does the same behaviour as the above one, but faster
        for i in range(0, m - self.kernel_size + 1, self.stride):
            for j in range(0, n - self.kernel_size + 1, self.stride):
                tmp = np.expand_dims(self.w, axis=0)
                tmp = np.repeat(tmp, self.batch_size, axis=0)
                ii  = i//self.stride
                jj  = j//self.stride
                tmp = (tmp.transpose(2,3,4,0,1) * e[:,ii,jj,:]).transpose(3,4,0,1,2)
                tmp = np.sum(tmp, axis=1)
                a[:,i:i+kernel,j:j+kernel,:] += tmp
        
        padding = self.padding
        p_height, p_width, _ = self.prev_layer.shape
        e = a[:,padding:p_height+padding,padding:p_width+padding,:]

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
        height = int((p_height - pool_size + 1) // stride)
        width  = int((p_width  - pool_size + 1) // stride)
        shape = (height, width, depth)
        super().__init__(None,None,shape,0,prev_layer)
        
    def __compute_maxpooling_gradient(self, a, e):
        gradient = np.zeros(shape=a.shape)
        for s in range(a.shape[0]):
            for l in range(a.shape[3]):
                c = a[s,:,:,l]
                i, j = np.unravel_index(c.argmax(), c.shape)
                gradient[s,i,j,l] = e[s,l]

        return gradient
    
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
                d_A[:,i*s:i*s+p,j*s:j*s+p,:] += self.__compute_maxpooling_gradient(a, e[:,i,j,:])

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