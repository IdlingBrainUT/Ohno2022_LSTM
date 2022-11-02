import numpy as np

from .functions import sigmoid, tanh

class LSTM:

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        f = sigmoid(f)
        g = tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        c_next = f * c_prev + g * i
        h_next = o * tanh(c_next)
        # h_next = o * sigmoid(c_next)
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        tanh_c_next = tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        # sig_c_next = sigmoid(c_next)
        # ds = dc_next + (dh_next * o) * sig_c_next * (1 - sig_c_next)
        dc_prev = ds * f
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        # do = dh_next * sig_c_next
        dg = ds * i
        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)
        dA = np.hstack((df, dg, di, do))
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)
        return dx, dh_prev, dc_prev

class Affine:

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.cache = None

    def forward(self, x):
        W, b = self.params
        out = sigmoid(np.dot(x, W) + b)
        self.cache = (x, out)
        return out

    def backward(self, dout):
        x, out = self.cache
        W, b = self.params
        dout = dout * (1.0 - out) * out
        db = np.sum(dout, axis=0)
        dW = np.dot(x.T, dout)
        dx = np.dot(dout, W.T)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx