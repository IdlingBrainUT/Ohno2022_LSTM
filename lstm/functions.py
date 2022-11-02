import numpy as np

def sigmoid(x):
    is_zero = x < -100
    is_calc = (x >= -100) & (x <= 100)
    ret = np.ones_like(x)
    ret[is_zero] = 0
    ret[is_calc] = 1 / (1 + np.exp(-x[is_calc]))
    return ret

def tanh(x):
    is_min1 = x < -100
    is_calc = (x >= -100) & (x <= 100)
    ret = np.ones_like(x)
    ret[is_min1] = -1
    epx = np.exp(x[is_calc])
    emx = np.exp(-x[is_calc])
    ret[is_calc] = (epx - emx) / (epx + emx)
    return ret