import numpy as np

class Mean_Absolute_Error:
    def loss(self, y_hat, y):
        return np.abs(y_hat - y)
    def derivative(self, y_hat, y):
        return np.where(y_hat - y >= 0, 1, -1)
    def cost(self, y_hat, y):
        return np.sum(np.abs(y_hat - y))/y.shape[0]
    def evaluation(self, y_hat, y):
        return f"Cost: {self.cost(y_hat, y)}"
class Mean_Squared_Error:
    def loss(self, y_hat, y):
        return ((y_hat - y)**2)/2
    def derivative(self, y_hat, y):
        return (y_hat - y)
    def cost(self, y_hat, y):
        return np.sum((y_hat - y)**2)/(2*y.shape[0])
    def evaluation(self, y_hat, y):
        return f"Cost: {self.cost(y_hat, y)}"
class Cross_Entropy:
    def loss(self, y_hat, y):
        return -np.sum(y * np.log(y_hat))
    def derivative(self, y_hat, y):
        return y_hat - y
    def cost(self, y_hat, y):
        return -np.sum(y*np.log(y_hat))/y.shape[0]
    def accuracy(self, y_hat, y):
        return np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))/y.shape[0]
    def evaluation(self, y_hat, y):
        return f"Cost: {self.cost(y_hat, y)}; Accuracy: {self.accuracy(y_hat, y)}"
class Binary_Cross_Entropy:
    def loss(self, y_hat, y):
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    def derivative(self, y_hat, y):
        return -((y/y_hat) - (1 - y)/(1 - y_hat))
    def cost(self, y_hat, y):
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    def accuracy(self, y_hat, y):
        return np.sum(y_hat == y)/y.shape[0]
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
    params = 0
    batch_size = None
    def __init__(self, shape):
        self.shape = shape
    def load_input(self, input):
        self.a = input
        self.batch_size = input.shape[0]
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

    params = None
    batch_size = None
    def __init__(self, shape, activation, previous_layer):
        self.shape = shape
        self.activation = activation
        self.previous_layer = previous_layer
        previous_layer.next_layer = self

        self.w = np.random.rand(shape, self.previous_layer.shape) * np.sqrt(2/self.previous_layer.shape)
        self.b = np.random.rand(shape) * np.sqrt(2/self.previous_layer.shape)
        self.derivative_w = np.zeros(shape=self.w.shape)
        self.derivative_b = np.zeros(shape=self.b.shape)
        self.params = shape * self.previous_layer.shape
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

    #compute derivative of activation function
    def derivative_of_activation(self):
        derivative = None
        if self.activation == 'ReLU':
            derivative = np.where(self.z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            derivative = self.a * (1 - self.a)
        elif self.activation == 'tanh':
            derivative = 1 - self.a**2
        elif self.activation == 'softmax':
            derivative = np.zeros(shape=([self.a.shape[0], self.a.shape[1], self.a.shape[1]]))
            for i in range(self.a.shape[0]):
                derivative[i] = np.diagflat(self.a[i]) - np.outer(self.a[i], self.a[i])
        elif self.activation == 'LeakyReLU':
            derivative = np.where(self.z > 0, 1, 0.1)
        elif self.activation == 'linear':
            derivative = np.ones(shape=self.shape)
        else:
            raise ValueError('Invalid activation function')

        return derivative
    def feed_forward(self):
        self.batch_size = self.previous_layer.batch_size
        self.z = np.dot(self.w, self.previous_layer.a.T).T + self.b
        self.activate()
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def back_propagation(self, e):
        derivative = self.derivative_of_activation()
        if self.activation == 'softmax':
            pass
        else:
            e = e * derivative
        for i in range(e.shape[0]):
            self.derivative_w += np.outer(e[i], self.previous_layer.a[i])

        self.derivative_b = np.sum(e, axis=0)

        e = np.dot(self.w.T, e.T).T

        #previous layer call back propagation
        self.previous_layer.back_propagation(e)
    def update_parameter(self, learning_rate):
        #update W and b
        self.w -= learning_rate * self.derivative_w / self.batch_size
        self.b -= learning_rate * self.derivative_b / self.batch_size

        #set derivative to zero
        self.derivative_w = np.zeros(shape=self.derivative_w.shape)
        self.derivative_b = np.zeros(shape=self.derivative_b.shape)
        
        self.previous_layer.update_parameter(learning_rate)
    
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
        self.get_model_parameters()
    def get_model_parameters(self):
        layer = self.input
        while layer:
            self.total_parameters += layer.params
            layer = layer.next_layer
        
    #add loss function and learning rate
    def compile(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate

    #train the model
    def fit(self, x_train, y_train, batch_size=None, epochs=None, iters=100, verbose=1):
        n_samples = x_train.shape[0]
        if batch_size == None:
            batch_size = n_samples

        batch_num = int(np.ceil(n_samples/batch_size))
        
        if epochs is not None:
            iters = epochs * batch_num

        for iter in range(iters):
            if iter % batch_num == 0:
                indices = np.random.permutation(np.arange(n_samples)).astype('int32')
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            low  = int(iter % batch_num) * batch_size
            high = min(low + batch_size, n_samples)
            x_batch = x_train[low:high]
            y_batch = y_train[low:high]

            if high - low == 1:
                x_batch = np.expand_dims(x_batch, axis=0)
                y_batch = np.expand_dims(y_batch, axis=0)

            self.input.load_input(x_batch)
            self.input.feed_forward()
            y_hat = self.output.a
            y = y_batch
            e = self.loss.derivative(y_hat, y)
            self.output.back_propagation(e)

            if verbose == 1:
                if epochs is None:
                    if (iter + 1) % 50 == 0:
                        y_predict = self.predict(x_train)
                        s = 'iter: ' + str(iter+1) + ' ' + self.loss.evaluation(y_predict, y_train)

                        print(s)
                else:
                    if (iter + 1) % batch_num == 0:
                        y_predict = self.predict(x_train)
                        s = 'epoch: ' + str(int((iter + 1) / batch_num)) + ' ' + self.loss.evaluation(y_predict, y_train)

                        print(s)
            
            elif verbose == 0:
                pass

            self.output.update_parameter(self.learning_rate)

    def predict(self, X_test):
        self.input.load_input(X_test)
        self.input.feed_forward()
        y_hat = self.output.a
        return y_hat
    
    def evaluate(self, X_test, Y_test, verbose=1):
        y_hat = self.predict(X_test)
        if verbose == 0:
            pass
        elif verbose == 1:
            print(self.loss.evaluation(y_hat, Y_test))
        
        loss = self.loss.loss(y_hat, Y_test)
        accuracy = self.loss.accuracy(y_hat, Y_test)

        return loss, accuracy

def processing(Y, n):
    y = np.zeros((Y.shape[0], n))
    for i in range(Y.shape[0]):
        y[i, int(Y[i])] = 1
    return y

def main():
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
    x_train = x_train.numpy()
    x_test  = x_test.numpy()

    y_train = processing(y_train, 10)
    y_test = processing(y_test, 10)

    input = Input(shape=28*28)
    layer = Layer(shape=256, activation='ReLU', previous_layer=input)
    layer = Layer(shape=64 , activation='ReLU', previous_layer=layer)
    output = Layer(shape=10, activation='softmax', previous_layer=layer)
    model = Model(input=input, output=output)
    model.compile(loss=Loss.CrossEntropy, learning_rate=0.01)

    start = time()
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    model.evaluate(x_test, y_test)
    end = time()
    print(end-start)

if __name__ == '__main__':
    main()