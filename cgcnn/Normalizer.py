import numpy as np
import sklearn.preprocessing as skpre
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
import copy
class Normalizer():
    def __init__(self, x):
        self.matrix = np.array(x, dtype=float)
        # print(self.matrix.shape)

    def minmax_normalize(self, x=None):  # 定义函数，对x进行归一化
        scaler = skpre.MinMaxScaler()
        if x:
            return scaler.fit_transform(x, x)
        else:
            return scaler.fit_transform(self.matrix, self.matrix)

    def absmax_normalize(self, x=None):
        scaler = skpre.MaxAbsScaler()
        if x:
            return scaler.fit_transform(x, x)
        else:
            return scaler.fit_transform(self.matrix, self.matrix)

    def minmax_minus1to1_normalize(self, x=None):
        if x:
            mean = np.array([0.5] * x.shape[1], dtype=float).reshape(1, -1)
            return np.true_divide(self.minmax_normalize(x) - mean, mean)
        else:
            mean = np.array([0.5] * self.matrix.shape[1], dtype=float).reshape(1, -1)
            return np.true_divide(self.minmax_normalize() - mean, mean)

    def standard_normalize(self, x=None):
        scaler = skpre.StandardScaler()
        if x:
            return scaler.fit_transform(x, x)
        else:
            return scaler.fit_transform(self.matrix, self.matrix)

    def reverse_minmax_minus1to1_normalize(self, src_x=None, x: np.array = None):
        assert src_x is not None, 'source matrix should not be None'
        min = np.min(src_x, 0)
        max = np.max(src_x, 0)
        if x is not None:
            return x * (max - min) + min
        else:
            return self.matrix * (max - min) + min

    @staticmethod
    def lambda_max(arr, axis=None, key=None, keepdims=False):
        if callable(key):
            idxs = np.argmax(key(arr), axis)
            if axis is not None:
                idxs = np.expand_dims(idxs, axis)
                result = np.take_along_axis(arr, idxs, axis)
                if not keepdims:
                    result = np.squeeze(result, axis=axis)
                return result
            else:
                return arr.flatten()[idxs]
        else:
            return np.amax(arr, axis)

    def inverse_absmax_normalize(self, src_x=None, x: np.array = None):
        # print(x*self.lambda_max(src_x,0,key=np.abs))
        if x is not None:
            return x * np.max(np.abs(src_x), 0)
        else:
            return self.matrix * (np.max(np.abs(src_x), 0))
