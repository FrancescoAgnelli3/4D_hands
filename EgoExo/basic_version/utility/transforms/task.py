import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class task(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        task = data["take_name"]
        if "basketball" in task:
            task = 0
        elif "dance" in task or "Dance" in task:
            task = 1
        elif "bike" in task:
            task = 2
        elif "music" in task:
            task = 3
        elif "covid" in task:
            task = 4
        elif "cooking" in task:
            task = 5
        elif "bouldering" in task:
            task = 6
        elif "Guitar" in task:
            task = 7
        elif "Violin" in task:
            task = 8
        elif "Piano" in task:
            task = 9
        elif "soccer" in task:
            task = 10
        elif "cpr" in task:
            task = 11
        data.task = task

        return data