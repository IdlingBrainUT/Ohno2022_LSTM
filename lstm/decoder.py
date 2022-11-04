import numpy as np

from .layers import LSTM, Affine

class Decoder:

    def __init__(self, params):
        self.LSTM_params = params[:3]
        self.Affine_params = params[3:]
        self.h_prev = np.zeros((1, params[1].shape[0]))
        self.c_prev = np.zeros((1, params[1].shape[0]))
        self.lstm = LSTM(*self.LSTM_params)
        self.affine = Affine(*self.Affine_params)

    def forward(self, xs):
        xs = xs.reshape(1, -1)
        hs, cs = self.lstm.forward(xs, self.h_prev, self.c_prev)
        self.h_prev = hs
        self.c_prev = cs
        out = self.affine.forward(hs)
        return out[0]

    def predict(self, x):
        return np.array([self.forward(xi) for xi in x])

    def reset(self):
        self.h_prev = np.zeros_like(self.h_prev)
        self.c_prev = np.zeros_like(self.c_prev)


