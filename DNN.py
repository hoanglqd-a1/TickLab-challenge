import numpy as np
from time import time

import Loss, Layer, Optimizers

#Input Layer
class Input:
    a = None
    shape = None

    next_layer = None

    params = 0
    batch_size = None
    def __init__(self, shape):
        self.shape = shape
    def load_input(self, input):
        self.a = input
    def feed_forward(self):
        if type(self.next_layer).__name__ == type(Layer.Conv2D).__name__ and self.a.ndim == 3:
            self.a = np.expand_dims(self.a, axis=3)

        self.batch_size = self.a.shape[0]
        self.next_layer.feed_forward()
    def back_propagation(self, *argv):
        return

class Model:
    input = None
    output = None
    loss = None
    total_parameters = 0
    #add input and output layer
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.__get_model_parameters()
    def __get_model_parameters(self):
        layer = self.input
        while layer:
            self.total_parameters += layer.params
            layer = layer.next_layer
        
    #add loss function and learning rate
    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        layer = self.input
        while layer:
            layer.optimizer = optimizer.__copy__()
            layer = layer.next_layer

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
                    if (iter + 1) % 1 == 0:
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
    from tensorflow.keras.datasets import cifar10
    import os 
    from time import time
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float64") / 255.0
    x_test  = x_test.astype('float64') / 255.0
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    x_train = (x_train.numpy())
    x_test = (x_test.numpy())

    y_train = y_train.reshape(-1)
    y_test  = y_test.reshape(-1)
    y_train = processing(y_train, 10)
    y_test  = processing(y_test , 10)

    x_train = x_train[:1000]
    x_test  = x_test[:50]
    y_train = y_train[:1000]
    y_test  = y_test[:50]

    input = Input(shape=(32, 32, 3))
    conv2d1 = Layer.Conv2D(filters=16, kernel_size=3, padding=1, activation='LeakyReLU', prev_layer=input)
    maxpooling1 = Layer.MaxPool2D(pool_size=2, prev_layer=conv2d1)
    conv2d2 = Layer.Conv2D(filters=32, kernel_size=3, padding=1, activation='LeakyReLU', prev_layer=maxpooling1)
    maxpooling2 = Layer.MaxPool2D(pool_size=2, prev_layer=conv2d2)
    flatten = Layer.Flatten(prev_layer=maxpooling2)
    layer1 = Layer.Layer(shape=64, activation='LeakyReLU', prev_layer=flatten)
    output = Layer.Layer(shape=10, activation='softmax', prev_layer=layer1)
    model = Model(input=input, output=output)
    model.compile(loss=Loss.CrossEntropy, optimizer=Optimizers.SGD(learning_rate=0.001, momentum=0.8))

    start = time()
    model.fit(x_train=x_train, y_train=y_train, batch_size=8, epochs=50)
    end   = time()
    print(end - start)

if __name__ == '__main__':
    main()  