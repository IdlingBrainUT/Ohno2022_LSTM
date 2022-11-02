import cupy as np
import pickle

class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump([p.astype(np.float16) for p in self.params], f)

    def load_params(self, file_name=None):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for i, param in enumerate(self.params):
            param[...] = params[i]