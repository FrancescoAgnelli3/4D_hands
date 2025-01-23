import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import controldiffeq

class SplineCoeff(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, data: Data) -> Data:
        land = data.land.permute(1,0,2).float()
        spline_coeff = controldiffeq.natural_cubic_spline_coeffs(data.times.float(), land)
        data.spline_coeff = spline_coeff
        # spline = controldiffeq.NaturalCubicSpline(data.times, spline_coeff)
        # data.spline = spline

        return data
