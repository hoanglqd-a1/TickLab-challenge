import cupy as cp

class Loss:
    class Mean_Absolute_Error:
        def loss(self, y_hat, y):
            return cp.abs(y_hat - y)
        def derivative(self, y_hat, y):
            return cp.where(y_hat - y >= 0, 1, -1)
        def cost(self, y_hat, y):
            return cp.sum(cp.abs(y_hat - y))/y.shape[0]
        def evaluation(self, y_hat, y):
            return f"Cost: {self.cost(y_hat, y)}"
    class Mean_Squared_Error:
        def loss(self, y_hat, y):
            return ((y_hat - y)**2)/2
        def derivative(self, y_hat, y):
            return (y_hat - y)
        def cost(self, y_hat, y):
            return cp.sum((y_hat - y)**2)/(2*y.shape[0])
        def evaluation(self, y_hat, y):
            return f"Cost: {self.cost(y_hat, y)}"
    class Cross_Entropy:
        def loss(self, y_hat, y):
            return -cp.sum(y * cp.log(y_hat))
        def derivative(self, y_hat, y):
            return y_hat - y
        def cost(self, y_hat, y):
            return -cp.sum(y*cp.log(y_hat))/y.shape[0]
        def accuracy(self, y_hat, y):
            return cp.sum(cp.argmax(y_hat, axis=1) == cp.argmax(y, axis=1))/y.shape[0]
        def evaluation(self, y_hat, y):
            return f"Cost: {self.cost(y_hat, y)}; Accuracy: {self.accuracy(y_hat, y)}"
    class Binary_Cross_Entropy:
        def loss(self, y_hat, y):
            return -cp.sum(y * cp.log(y_hat) + (1 - y) * cp.log(1 - y_hat))
        def derivative(self, y_hat, y):
            return -((y/y_hat) - (1 - y)/(1 - y_hat))
        def cost(self, y_hat, y):
            return -cp.sum(y * cp.log(y_hat) + (1 - y) * cp.log(1 - y_hat))
        def accuracy(self, y_hat, y):
            return cp.sum(y_hat == y)/y.shape[0]
        def evaluation(self, y_hat, y):
            return f"Cost: {self.cost(y_hat, y)}; Accuracy: {self.accuracy(y_hat, y)}"
        
    MAE = Mean_Absolute_Error()
    MSE = Mean_Squared_Error()
    CrossEntropy = Cross_Entropy()
    BinaryCrossEntropy = Binary_Cross_Entropy()

#Input Layer
class Input:
    a = None
    shape = None
    next_layer = None
    previous_layer = None
    parameters_num = 0
    def __init__(self, shape):
        self.shape = shape
    def load_input(self, input):
        self.a = input
    def feed_forward(self):
        self.next_layer.feed_forward()
    def back_propagation(self, *argv):
        return
    def update_parameter(self, *argv):
        return

class Layer:
    z = None
    w = None
    b = None
    a = None
    #activation function
    activation = None
    shape = None
    previous_layer = None
    next_layer = None

    #partial derivative of loss function to W and b
    derivative_w = None
    derivative_b = None

    parameters_num = None
    def __init__(self, shape, activation, previous_layer):
        self.shape = shape
        self.activation = activation
        self.previous_layer = previous_layer
        previous_layer.next_layer = self
        self.w = cp.random.rand(shape, self.previous_layer.shape)
        self.b = cp.random.rand(shape)
        self.derivative_w = cp.zeros(shape=[shape, self.previous_layer.shape])
        self.derivative_b = cp.zeros(shape=shape)
        self.parameters_num = shape * self.previous_layer.shape
    def activate(self):
        if self.activation == 'ReLU':
            self.a = cp.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.a = 1/(1 + cp.exp(-self.z))
        elif self.activation == 'tanh':
            self.a = cp.tanh(self.z)
        elif self.activation == 'softmax':
            if self.z.ndim == 1:
                self.z = cp.array([self.z])
            z = (self.z.T - cp.max(self.z, axis=1)).T
            self.a = (cp.exp(z).T / cp.sum(cp.exp(z), axis=1)).T
            self.a = cp.maximum(self.a, 1e-6)
        elif self.activation == 'LeakyReLU':
            self.a = cp.maximum(self.z/10, self.z)
        elif self.activation == 'linear':
            self.a = self.z
        else:
            raise ValueError('Invalid activation function')

    #compute derivative of activation function
    def derivative_of_activation(self):
        derivative = None
        if self.activation == 'ReLU':
            derivative = cp.where(self.z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            derivative = self.a * (1 - self.a)
        elif self.activation == 'tanh':
            derivative = 1 - self.a**2
        elif self.activation == 'softmax':
            derivative = cp.zeros(shape=([self.a.shape[0], self.a.shape[1], self.a.shape[1]]))
            for i in range(self.a.shape[0]):
                derivative[i] = cp.diagflat(self.a[i]) - cp.outer(self.a[i], self.a[i])
        elif self.activation == 'LeakyReLU':
            derivative = cp.where(self.z > 0, 1, 0.1)
        elif self.activation == 'linear':
            derivative = cp.ones(shape=self.shape)
        else:
            raise ValueError('Invalid activation function')

        return derivative
    def feed_forward(self):
        self.z = cp.dot(self.w, self.previous_layer.a.T).T
        self.z = self.z + self.b
        self.activate()
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def back_propagation(self, e, learning_rate):
        derivative = self.derivative_of_activation()
        #print('d_acti:', derivative[0], e[0])
        if self.activation == 'softmax':
            pass
        else:
            e = e * derivative
        #print(e[0])
        self.derivative_w = cp.zeros(shape=[e.shape[0], self.w.shape[0], self.w.shape[1]])
        for i in range(e.shape[0]):
            self.derivative_w[i] = learning_rate * cp.outer(e[i], self.previous_layer.a[i])
        self.derivative_w = cp.sum(self.derivative_w, axis=0)

        self.derivative_b = learning_rate * e
        self.derivative_b = cp.sum(self.derivative_b, axis=0)

        #next layer call back propagation
        e = cp.dot(self.w.T, e.T).T

        #previous layer call back propagation
        self.previous_layer.back_propagation(e, learning_rate)
    def update_parameter(self):
        #update W and b
        self.w = self.w - self.derivative_w
        self.b = self.b - self.derivative_b

        #set derivative to zero
        self.derivative_w = cp.zeros(shape=self.derivative_w.shape)
        self.derivative_b = cp.zeros(shape=self.derivative_b.shape)
        self.previous_layer.update_parameter()
    

class Model:
    input = None
    output = None
    loss = None
    learning_rate = None
    total_parameters = 0
    #add input and output layer
    def __init__(self, input, output):
        self.input = input
        self.output = output
    def get_model_parameters(self):
        layer = self.input
        while layer:
            self.total_parameters += layer.parameters_num
            layer = layer.next_layer
        
    #add loss function and learning rate
    def compile(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate

    #train the model
    def fit(self, x_train, y_train, batch_size=None, iters=10):
        n_samples = x_train.shape[0]
        if batch_size == None:
            batch_size = n_samples

        for iter in range(iters):
            indices = cp.random.permutation(cp.arange(n_samples))[:batch_size]
            x_batch = x_train[indices]
            y_batch = y_train[indices]

            self.input.load_input(x_batch)
            self.input.feed_forward()
            y_hat = self.output.a
            y = y_batch
            e = self.loss.derivative(y_hat, y)/batch_size
            self.output.back_propagation(e, self.learning_rate)

            if (iter + 1) % 50 == 0:
                y_predict = self.predict(x_train)
                s = 'iter:' + str(iter+1) + self.loss.evaluation(y_predict, y_train)

                print(s)
            
            self.output.update_parameter()

    def predict(self, X_test):
        self.input.load_input(X_test)
        self.input.feed_forward()
        y_hat = self.output.a
        return y_hat
    
    def evaluate(self, X_test, Y_test):
        y_hat = self.predict(X_test)
        print(self.loss.evaluation(y_hat, Y_test))

def processing(Y, n):
    y = cp.zeros((Y.shape[0], n))
    for i in range(Y.shape[0]):
        y[i, int(Y[i])] = 1
    return y

"""def main():
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    import os 
    from time import time
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 28*28).astype('float32') / 255.0
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    x_train = cp.asarray(x_train.numpy())
    x_test = cp.asarray(x_test.numpy())

    y_train = processing(y_train, 10)
    y_test = processing(y_test, 10)

    input = Input(shape=28*28)
    layer = Layer(shape=256, activation='ReLU', previous_layer=input)
    output = Layer(shape=10, activation='softmax', previous_layer=layer)
    model = Model(input=input, output=output)
    model.compile(loss=Loss.CrossEntropy, learning_rate=0.001)

    start = time()
    model.fit(x_train, y_train, batch_size=32, iters=4000)
    model.evaluate(x_test, y_test)
    end = time()
    print(end-start)

if __name__ == '__main__':
    main()"""