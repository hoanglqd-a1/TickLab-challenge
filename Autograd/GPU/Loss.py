import cupyautograd as ag
import numpy as np

class Mean_Absolute_Error:
    def cost(self, y_hat, y):
        return ag.sum(ag.abs(y_hat - y))/y.shape[0]
    def evaluation(self, y_hat, y):
        return "Cost: {:.4f}".format(self.cost(y_hat, y))
class Mean_Squared_Error:
    def cost(self, y_hat, y):
        return ag.sum((y_hat - y)**2)/(2*y.shape[0])
    def evaluation(self, y_hat, y):
        return "Cost: {:.4f}".format(self.cost(y_hat, y))
class Cross_Entropy:
    def cost(self, y_hat: ag.Var, y: ag.Var):
        return -ag.sum(y*ag.log(y_hat))/y.shape[0]
    def accuracy(self, y_hat: ag.Var, y: ag.Var):
        return ag.sum(ag.argmax(y_hat, axis=1) == ag.argmax(y, axis=1))/y.shape[0]
    def evaluation(self, y_hat: ag.Var, y:ag.Var):
        return "Cost: {:.4f}; Accuracy: {:.4f}".format(self.cost(y_hat, y).value, self.accuracy(y_hat, y).value)
class Binary_Cross_Entropy:
    def cost(self, y_hat, y):
        return -ag.sum(y * ag.log(y_hat) + (1 - y) * ag.log(1 - y_hat))
    def accuracy(self, y_hat, y):
        return ag.sum(y_hat == y)/y.shape[0]
    def evaluation(self, y_hat, y):
        return "Cost: {:.4f}; Accuracy: {:.4f}".format(self.cost(y_hat, y), self.accuracy(y_hat, y))
    
MAE = Mean_Absolute_Error()
MSE = Mean_Squared_Error()
CrossEntropy = Cross_Entropy()
BinaryCrossEntropy = Binary_Cross_Entropy()