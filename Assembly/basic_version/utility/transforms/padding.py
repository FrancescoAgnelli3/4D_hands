import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class padding(BaseTransform):
    def __init__(self, length):
        self.length = length
    def __call__(self, data: Data) -> Data:
        N, M, T = data.right_land.shape
        padded_tensor = torch.full((self.length, M, T), float('nan'))
        padded_tensor[:N, :, :] = data.right_land
        data.right_land = padded_tensor

        N, M, T = data.left_land.shape
        padded_tensor = torch.full((self.length, M, T), float('nan'))
        padded_tensor[:N, :, :] = data.left_land
        data.left_land = padded_tensor

        N, M, T = data.land.shape
        padded_tensor = torch.full((self.length, M, T), float('nan'))
        padded_tensor[:N, :, :] = data.land
        data.land = padded_tensor

        data.times = torch.arange(self.length)

        return data
