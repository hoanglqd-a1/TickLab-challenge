import numpy as np
import cupy as cp
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
        if isinstance(self.next_layer, Layer.Conv2D) and self.a.ndim == 3:
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
    def fit(self, x_train, y_train, x_validate=None, y_validate=None, batch_size=None, epochs=None, iters=100, verbose=1):
        n_samples = x_train.shape[0]
        if batch_size == None:
            batch_size = n_samples

        batch_num = int(np.ceil(n_samples/batch_size))
        
        if epochs is not None:
            iters = epochs * batch_num

        start = time()
        for iter in range(iters):
            if iter % batch_num == 0:
                indices = np.random.permutation(np.arange(n_samples)).astype('int32')
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            low  = int(iter % batch_num) * batch_size
            high = min(low + batch_size, n_samples)
            x_batch = x_train[low:high]
            y_batch = y_train[low:high]

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
                        print(f'Iter: {iter}')
                        out = "training: " + self.loss.evaluation(y_predict, y_train)
                        if x_validate is not None:
                            y_predict = self.predict(x_validate)
                            out += "validation: " + self.loss.evaluation(y_predict, y_validate)
                        print(out)
                        print('time:',time() - start)
                        start = time()
                else:
                    if (iter + 1) % batch_num == 0:
                        y_predict = self.predict(x_train)
                        print(f'Epoch: {(iter + 1)//batch_num}')
                        out = "Training. " + self.loss.evaluation(y_predict, y_train)
                        if x_validate is not None:
                            y_predict = self.predict(x_validate)
                            out += "; Validation. " + self.loss.evaluation(y_predict, y_validate)
                        print(out)
                        print('time:',time() - start)
                        start = time()
            
            elif verbose == 0:
                pass

    def predict(self, X_test, batch_size=None):
        if batch_size == None:
            batch_size = min(258, X_test.shape[0])
        y_hat = np.zeros(shape=[X_test.shape[0], self.output.shape])
        for i in range(0, X_test.shape[0], batch_size):
            end = min(i + batch_size, X_test.shape[0])
            x_batch = X_test[i:end]
            if end - i == 1:
                x_batch = np.expand_dims(x_batch, axis=0)
            self.input.load_input(x_batch)
            self.input.feed_forward()
            y_hat[i:end] = self.output.a
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
    x_train = x_train.astype("float64") / 255.0
    x_test  = x_test.astype('float64') / 255.0
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    x_train = (x_train.numpy())
    x_test = (x_test.numpy())

    #x_train = x_train.reshape(60000, -1)
    #x_test  = x_test.reshape(10000, -1)

    y_train = y_train.reshape(-1)
    y_test  = y_test.reshape(-1)
    y_train = processing(y_train, 10)
    y_test  = processing(y_test , 10)

    x_validate = x_test[:5000]
    y_validate = y_test[:5000]
    x_test = x_test[5000:]
    y_test = y_test[5000:]

    input = Input(shape=(28, 28, 1))
    conv2d1 = Layer.Conv2D(filters=16, kernel_size=3, padding=1, activation='ReLU', prev_layer=input)
    maxpooling1 = Layer.MaxPool2D(pool_size=2, prev_layer=conv2d1)
    conv2d2 = Layer.Conv2D(filters=32, kernel_size=3, padding=1, activation='ReLU', prev_layer=maxpooling1)
    maxpooling2 = Layer.MaxPool2D(pool_size=2, prev_layer=conv2d2)
    flatten = Layer.Flatten(prev_layer=maxpooling2)
    layer1 = Layer.Layer(shape=128, activation='ReLU', prev_layer=flatten)
    output = Layer.Layer(shape=10, activation='softmax', prev_layer=layer1)
    model = Model(input=input, output=output)
    model.compile(loss=Loss.CrossEntropy, optimizer=Optimizers.Adam(learning_rate=0.005))
    model.fit(x_train, y_train, x_validate, y_validate, batch_size=32, epochs=5)
    model.evaluate(x_test, y_test)

if __name__ == '__main__':  
    main()  