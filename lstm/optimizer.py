import numpy as np

class SGD:
    def __init__(self, lr=1):
        self.lr = lr
        
    def update(self, params, grads):
        # print(cp.abs(params[0]).max(), cp.abs(grads[0]).max())
        for i in range(len(params)):
            # if i < 3:
            params[i] -= self.lr * grads[i]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]