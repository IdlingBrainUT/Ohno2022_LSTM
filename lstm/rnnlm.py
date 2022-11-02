import numpy as np

from .time_layers import TimeAffine, TimeLSTM, TimeSquareSumLoss
from .base_model import BaseModel

class Rnnlm(BaseModel):
    def __init__(self, input_size, output_size=5, rate=[1, 1, 1]):
        rn = np.random.randn

        # 重みの初期化
        lstm_Wx = (rn(input_size, 4 * output_size) / np.sqrt(input_size)).astype('f')
        lstm_Wh = (rn(output_size, 4 * output_size) / np.sqrt(output_size)).astype('f')
        lstm_b = np.zeros(4 * output_size).astype('f')
        affine_W = (rn(output_size, output_size) / np.sqrt(output_size)).astype('f')
        # affine_W = np.zeros((output_size, output_size)).astype('f')
        # for i in range(output_size):
        #     affine_W[i, i] = 10
        affine_b = np.zeros(output_size).astype('f')

        # レイヤの生成
        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSquareSumLoss(rate)
        self.lstm_layer = self.layers[0]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self.grads = []
        for layer in self.layers:
            self.grads += layer.grads
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()