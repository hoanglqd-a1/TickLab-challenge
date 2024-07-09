import numpy as np

class Loss:
    class MAE:
        def loss(self, y_hat, y):
            return np.abs(y_hat - y)
        def derivative(self, y_hat, y):
            return np.where(y_hat - y >= 0, 1, -1)
        def cost(self, y_hat, y):
            return np.sum(np.abs(y_hat - y))/y.shape[0]
        def evalute(self, y_hat, y):
            print('Cost:', self.cost(y_hat, y))
    class MSE:
        def loss(self, y_hat, y):
            return ((y_hat - y)**2)/2
        def derivative(self, y_hat, y):
            return (y_hat - y)
        def cost(self, y_hat, y):
            return np.sum((y_hat - y)**2)/(2*y.shape[0])
        def evaluate(self, y_hat, y):
            print('Cost:', self.cost(y_hat, y))
    class CrossEntropy:
        def loss(self, y_hat, y):
            return -np.sum(y * np.log(y_hat))
        def derivative(self, y_hat, y):
            return -(y/y_hat)
        def cost(self, y_hat, y):
            return -np.sum(y*np.log(y_hat))
        def accuracy(self, y_hat, y):
            return np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))/y.shape[0]
        def evaluate(self, y_hat, y):
            print('Cost:', self.cost(y_hat, y), 'Accuracy:', self.accuracy(y_hat, y))
    class BinaryCrossEntropy:
        def loss(self, y_hat, y):
            return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        def derivative(self, y_hat, y):
            return -((y/y_hat) - (1 - y)/(1 - y_hat))
        def cost(self, y_hat, y):
            return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        def accuracy(self, y_hat, y):
            return np.sum(y_hat == y)/y.shape[0]
        def evaluate(self, y_hat, y):
            print('Cost:', self.cost(y_hat, y), 'Accuracy:', self.accuracy(y_hat, y))

#Input Layer
class Input:
    a = None
    shape = None
    next_layer = None
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
    def __init__(self, shape, activation, previous_layer):
        self.shape = shape
        self.activation = activation
        self.previous_layer = previous_layer
        previous_layer.next_layer = self
        self.w = np.random.rand(shape, self.previous_layer.shape)
        self.b = np.random.rand(shape)
        self.derivative_w = np.zeros(shape=[shape, self.previous_layer.shape])
        self.derivative_b = np.zeros(shape=shape)
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
        elif self.activation == 'linear':
            derivative = np.ones(shape=self.shape)
        else:
            raise ValueError('Invalid activation function')

        return derivative
    def feed_forward(self):
        self.z = np.dot(self.w, self.previous_layer.a.T).T
        self.z = self.z + self.b
        self.activate()
        if self.next_layer != None:
            self.next_layer.feed_forward()
    def back_propagation(self, e, learning_rate):
        derivative = self.derivative_of_activation()
        if self.activation == 'softmax':
            try:
                for i in range(e.shape[0]):
                    e[i] = np.dot(derivative[i], e[i])
            except:
                print('error')
        else:
            e = e * derivative
        self.derivative_w = np.zeros(shape=[e.shape[0], self.w.shape[0], self.w.shape[1]])
        for i in range(e.shape[0]):
            self.derivative_w[i] = learning_rate * np.outer(e[i], self.previous_layer.a[i])
        self.derivative_w = np.sum(self.derivative_w, axis=0)

        self.derivative_b = learning_rate * e
        self.derivative_b = np.sum(self.derivative_b, axis=0)

        #next layer call back propagation
        a = np.zeros(shape=[e.shape[0], self.previous_layer.shape])
        for i in range(e.shape[0]):
            a[i] = np.dot(self.w.T, e[i])

        #previous layer call back propagation
        e = a
        self.previous_layer.back_propagation(e, learning_rate)
    def update_parameter(self):
        #update W and b
        self.w = self.w - self.derivative_w
        self.b = self.b - self.derivative_b

        #set derivative to zero
        self.derivative_w = np.zeros(shape=self.derivative_w.shape)
        self.derivative_b = np.zeros(shape=self.derivative_b.shape)
        self.previous_layer.update_parameter()

class Model:
    input = None
    output = None
    loss = None
    learning_rate = None

    #add input and output layer
    def __init__(self, input, output):
        self.input = input
        self.output = output
    #add loss function and learning rate
    def compile(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate

    #train the model
    def fit(self, x_train, y_train, batch_size=None, epochs=10):
        n_samples = x_train.shape[0]
        if batch_size == None:
            batch_size = n_samples

        n_batches = np.ceil(n_samples/batch_size)
        for epoch in range(1, epochs+1):
            cost = 0.0
            batch = int(epoch % n_batches)
            head = batch * batch_size
            tail = min((batch+1) * batch_size, n_samples)
            
            x_batch = x_train[head:tail]
            y_batch = y_train[head:tail]

            self.input.load_input(x_batch)
            self.input.feed_forward()
            y_hat = self.output.a
            y = y_batch
            cost += self.loss.loss(y_hat, y)
            e = self.loss.derivative(y_hat, y)/batch_size
            self.output.back_propagation(e, self.learning_rate)

            if epoch % 50 == 0:
                y_hat = self.predict(x_train)
                s = 'epoch:' + str(epoch) + ', cost:' + str(self.loss.cost(y_hat, y_train))
                if type(self.loss).__name__ == type(Loss.CrossEntropy()).__name__:
                    s += ', accuracy:' + str(self.loss.accuracy(y_hat, y_train))

                print(s)
            
            self.output.update_parameter()

    def predict(self, X_test):
        self.input.load_input(X_test)
        self.input.feed_forward()
        y_hat = self.output.a
        return y_hat
    
    def evaluate(self, X_test, Y_test):
        y_hat = self.predict(X_test)
        self.loss.evaluate(y_hat, Y_test)

def processing(Y, n):
    y = np.zeros((Y.shape[0], n))
    for i in range(Y.shape[0]):
        y[i, int(Y[i])] = 1
    return y
    
def main():
    x_train = np.array([[1,2,3,4,5,6,7,8,9,10]])
    y_train = np.array([[1, 0, 0]])
    inputs = Input(shape=10)
    layer1 = Layer(shape=8, activation='ReLU', previous_layer=inputs)
    layer2 = Layer(shape=6, activation='ReLU', previous_layer=layer1)
    outputs = Layer(shape=3, activation='softmax', previous_layer=layer2)
    model = Model(input=inputs, output=outputs)
    model.compile(loss=Loss.CrossEntropy(), learning_rate=0.001)
    model.fit(x_train=x_train, y_train=y_train, batch_size=1, epochs=100)

if __name__ == '__main__':
    main()