import numpy as np

class Optimizer:
    learning_rate = None
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class SGD(Optimizer):
    momentum = 0.0
    change_W = 0.0
    change_b = 0.0
    def __init__(self, learning_rate, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
    def __copy__(self):
        return SGD(self.learning_rate, self.momentum)
    def compute_change(self,d_W,d_b):
        self.change_W = self.momentum * self.change_W + self.learning_rate * d_W
        self.change_b = self.momentum * self.change_b + self.learning_rate * d_b
        return self.change_W, self.change_b

class Adam(Optimizer):
    t = 0
    b1, b2 = 0.9, 0.999
    b1_t, b2_t = 1.0, 1.0
    m_w, v_w, m_b, v_b = 0.0, 0.0, 0.0, 0.0
    def __init__(self, learning_rate, b1=0.9, b2=0.999):
        super().__init__(learning_rate)
        self.b1 = b1
        self.b2 = b2
    def __copy__(self):
        return Adam(self.learning_rate, self.b1, self.b2)
    def compute_change(self,d_W,d_b):
        self.t += 1
        self.m_w = self.b1 * self.m_w + (1 - self.b1) * d_W
        self.v_w = self.b2 * self.v_w + (1 - self.b2) * np.square(d_W)
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        m_w_hat  = self.m_w / (1 - self.b1_t)
        v_w_hat  = self.v_w / (1 - self.b2_t)
        self.m_b = self.b1 * self.m_b + (1 - self.b1) * d_b
        self.v_b = self.b2 * self.v_b + (1 - self.b2) * np.square(d_b)
        m_b_hat  = self.m_b / (1 - self.b1_t)
        v_b_hat  = self.v_b / (1 - self.b2_t)
        change_W = self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
        change_b = self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
        return change_W, change_b
        
    