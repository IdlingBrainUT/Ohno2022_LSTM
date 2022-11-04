import numpy as np

from .functions import sigmoid
from .layers import LSTM

class TimeLSTM:

    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        n_hidden = self.params[1].shape[0]
        n_batch, n_time, _ = xs.shape
        self.layers = []
        hs = np.empty((n_batch, n_time, n_hidden), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((n_batch, n_hidden), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((n_batch, n_hidden), dtype='f')
        for t in range(n_time):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        n_input = self.params[0].shape[0]
        n_batch, n_time, _ = dhs.shape
        dxs = np.empty((n_batch, n_time, n_input), dtype='f')
        dh, dc = 0, 0
        grads = [0, 0, 0]
        for t in reversed(range(n_time)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None
        
class TimeAffine:

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.cache = None

    def forward(self, x):
        n_batch, n_time, n_hidden = x.shape
        W, b = self.params
        rx = x.reshape(n_batch*n_time, n_hidden)
        out = sigmoid(np.dot(rx, W) + b)
        self.cache = (x, out)
        return out.reshape(n_batch, n_time, n_hidden)

    def backward(self, dout):
        x, out = self.cache
        n_batch, n_time, n_hidden = x.shape
        W = self.params[0]
        dout = dout.reshape(n_batch*n_time, -1) * (1.0 - out) * out
        rx = x.reshape(n_batch*n_time, n_hidden)
        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class TimeSquareSumLoss:

    def __init__(self, rate):
        self.params, self.grads = [], []
        self.rate = rate
        self.cache = None

    def forward(self, xs, ts):
        n_batch, n_time, _ = xs.shape
        n_time_t = ts.shape[1]
        diff = np.zeros_like(xs)
        diff[:, -n_time_t:, :] = xs[:, -n_time_t:, :] - ts
        diff_null_xy = np.zeros((n_batch, n_time_t, 2))
        diff_null_xy[:, :, :] = ts[:, :, 3:] - (ts[:, :, 3:].sum(axis=0).sum(axis=0) / (n_batch * n_time_t))[np.newaxis, np.newaxis, :]
        scale = np.ones(5) * 0.25
        loss_raw = np.square(diff) / 2
        loss_raw /= scale[np.newaxis, np.newaxis, :] * 4
        mask = np.ones_like(loss_raw)
        mask[:, :-n_time_t, :] = 0
        mask[:, -n_time_t:, 2][ts[:, :, 1] == 0] = 0
        for i in range(3):
            mask[:, -n_time_t:, i][ts[:, :, i] == 1] *= self.rate[i]
        loss_sum = (loss_raw * mask).sum(axis=0).sum(axis=0)
        mask_sum = mask.sum(axis=0).sum(axis=0)
        mask_sum[mask_sum == 0] += 1
        loss = loss_sum / mask_sum
        self.cache = (mask, mask_sum, diff)
        return loss

    def backward(self, dout=1):
        mask, mask_sum, diff = self.cache
        dx = dout * diff
        dx *= mask
        return dx / mask_sum[np.newaxis, np.newaxis, :]