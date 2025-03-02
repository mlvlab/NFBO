import numpy as np
import torch


class NScaler(object):
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data
    def fit(self, data):
        pass

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        elif type(data) == torch.Tensor and type(self.mean) == torch.Tensor and data.device != self.mean.device:
            self.std = (self.std).to(data.device).type(data.dtype)
            self.mean = (self.mean).to(data.device).type(data.dtype)
        div = self.std
        return (data - self.mean) / torch.where(div < 1e-8, 1, div)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        elif type(data) == torch.Tensor and type(self.mean) == torch.Tensor and data.device != self.mean.device:
            self.std = (self.std).to(data.device).type(data.dtype)
            self.mean = (self.mean).to(data.device).type(data.dtype)
            
        return (data * self.std) + self.mean
    
    def fit(self, data):
        self.std = data.std(0)
        self.mean = data.mean(0)

class MinMaxScaler:
    """
    Standard the input
    """

    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        elif type(data) == torch.Tensor and type(self.min) == torch.Tensor and data.device != self.min.device:
            self.min = (self.min).to(data.device).type(data.dtype)
            self.max = (self.max).to(data.device).type(data.dtype)
        div = (self.max - self.min)
        return (data - self.min) / torch.where(div < 1e-8, 1, div)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        elif type(data) == torch.Tensor and type(self.min) == torch.Tensor and data.device != self.min.device:
            self.min = (self.min).to(data.device).type(data.dtype)
            self.max = (self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)
    
    def fit(self, data):
        self.min = data.min(0)
        self.max = data.max(0)
        if type(data) == torch.Tensor:
            self.min = self.min.values
            self.max = self.max.values

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        elif type(data) == torch.Tensor and type(self.min) == torch.Tensor and data.device != self.min.device:
            self.min = (self.min).to(data.device).type(data.dtype)
            self.max = (self.max).to(data.device).type(data.dtype)
        div = (self.max - self.min)
        return ((data - self.min) / torch.where(div < 1e-8, 1, div)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        elif type(data) == torch.Tensor and type(self.min) == torch.Tensor and data.device != self.min.device:
            self.min = (self.min).to(data.device).type(data.dtype)
            self.max = (self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min
    
    def fit(self, data):
        self.min = data.min(0)
        self.max = data.max(0)
        if type(data) == torch.Tensor:
            self.min = self.min.values
            self.max = self.max.values