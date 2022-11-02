from multiprocessing.sharedctypes import Value
import numpy as np
import time

class RnnlmTrainer:
    def __init__(self, model, optimizer, seed=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = None
        if seed is not None:
            np.random.seed(seed)
        self.time_idx = None
        self.params = []

    """
    def get_batch(self, x, t, batch_size, time_size):
        data_size, I = x.shape
        _, out_size = t.shape
        batch_x = np.empty((batch_size, time_size, I), dtype='f')
        batch_t = np.empty((batch_size, time_size, out_size), dtype='f')

        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for i, offset in enumerate(offsets):
            tmp = (offset + self.time_idx) % data_size
            if data_size - tmp >= time_size:
                batch_x[i, :, :] = x[tmp:tmp+time_size, :]
                batch_t[i, :, :] = t[tmp:tmp+time_size, :]
            else:
                tmp2 = data_size - tmp
                tmp3 = time_size - tmp2
                batch_x[i, :tmp2, :] = x[tmp:, :]
                batch_x[i, tmp2:, :] = x[:tmp3, :]
                batch_t[i, :tmp2, :] = t[tmp:, :]
                batch_t[i, tmp2:, :] = t[:tmp3, :]
        self.time_idx += time_size
        return batch_x, batch_t
    """

    def fit(self, xs, ts, max_epoch=10, max_iters=10, batch_size=10, 
            max_effect=150, min_effect=50, max_grad=None, save_params=False):
        if max_effect < min_effect:
            raise ValueError("max_effect is less than min_effect")
        data_size, input_size = xs.shape
        _, output_size = ts.shape
        window_size = int(data_size / batch_size)
        idx_size = window_size - max_effect + 1
        if idx_size <= 0:
            raise ValueError("idx_size is not positive")
        idx_arr = np.random.permutation(np.arange(idx_size))
        self.time_idx = 0
        self.loss_list = []
        model, optimizer = self.model, self.optimizer

        start_time = time.time()
        for epoch in range(max_epoch):
            total_loss = np.zeros(output_size)
            num_loss = 0
            for iter in range(max_iters):
                batch_x, batch_t = np.zeros((batch_size, max_effect, input_size)), np.zeros((batch_size, max_effect - min_effect, output_size))
                idx = idx_arr[self.time_idx]
                for batch in range(batch_size):
                    pos = batch*window_size+idx
                    batch_x[batch, :, :] = xs[pos:pos+max_effect, :]
                    batch_t[batch, :, :] = ts[pos+min_effect:pos+max_effect, :]
                
                model.reset_state()
                loss = model.forward(batch_x, batch_t)
                model.backward()
                if max_grad is not None:
                    clip_grads(model.grads, max_grad)
            
                params_pre = []
                for param in model.params:
                    params_pre.append(param.copy())

                loss_post = None
                lr = optimizer.lr
                # print("loss: ", loss)
                for _ in range(10):
                    optimizer.update(model.params, model.grads)
                    model.reset_state()
                    loss_post = model.forward(batch_x, batch_t)
                    # print("loss_post", loss_post)
                    if np.sum(loss * 1.05 >= loss_post) == len(loss):
                        break
                    for i, param in enumerate(params_pre):
                        model.params[i][...] = param[...]
                    # print("lr: ", optimizer.lr, " -> ", optimizer.lr * 0.1)
                    optimizer.lr *= 0.1
                optimizer.lr = lr

                total_loss += loss_post
                num_loss += 1
                self.time_idx = (self.time_idx + 1) % idx_size
                
            elapsed_time = time.time() - start_time
            mean_loss = total_loss / num_loss

            if save_params:
                params_save = []
                for param in model.params:
                    params_save.append(param.copy())
                self.params.append(params_save)

            print('epoch %d / %d | time %d[s] | whe: %.2f, lic: %.2f, rew: %.2f, x: %.2f, y: %.2f '
                      % (epoch, max_epoch, elapsed_time, mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3], mean_loss[4]))
            self.loss_list.append(mean_loss)

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate