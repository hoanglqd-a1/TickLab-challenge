import numpy as np

class Mean_Absolute_Error:
    def loss(self, y_hat, y):
        return np.abs(y_hat - y)
    def derivative(self, y_hat, y):
        return np.where(y_hat - y >= 0, 1, -1)
    def cost(self, y_hat, y):
        return np.sum(np.abs(y_hat - y))/y.shape[0]
    def evaluation(self, y_hat, y):
        return "Cost: {:.4f}".format(self.cost(y_hat, y))
class Mean_Squared_Error:
    def loss(self, y_hat, y):
        return ((y_hat - y)**2)/2
    def derivative(self, y_hat, y):
        return (y_hat - y)
    def cost(self, y_hat, y):
        return np.sum((y_hat - y)**2)/(2*y.shape[0])
    def evaluation(self, y_hat, y):
        return "Cost: {:.4f}".format(self.cost(y_hat, y))
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
        return "Cost: {:.4f}; Accuracy: {:.4f}".format(self.cost(y_hat, y), self.accuracy(y_hat, y))
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
        return "Cost: {:.4f}; Accuracy: {:.4f}".format(self.cost(y_hat, y), self.accuracy(y_hat, y))
    
MAE = Mean_Absolute_Error()
MSE = Mean_Squared_Error()
CrossEntropy = Cross_Entropy()
BinaryCrossEntropy = Binary_Cross_Entropy()