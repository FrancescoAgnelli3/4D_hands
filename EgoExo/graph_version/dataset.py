import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import open3d as o3d
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
import pymeshfix
import plotly.graph_objects as go
from torch_geometric.data.collate import collate
import json


class Ego4D(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")
    
    @property
    def processed_dir_slice(self):
        return osp.join(self.root, "processed_slice")

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [f for f in os.listdir(self.processed_dir) if f.endswith(".pt") and f.startswith("data")]

    def apply_transform_to_dataset(self, transform: BaseTransform, save: bool = True):
        """
        Apply a transform to each Data object in the dataset. The transform can be a T.Compose object. If save is True, the transformed Data objects will be saved to disk, overwriting the original ones.

        Args:
            transform (BaseTransform): The transform to apply to each Data object. It can be a T.Compose object.

        Returns:
            None
        """
        print(f"Applying transform {transform} to dataset...")
        for i in tqdm(range(len(self.processed_file_names))):
            try:
                data = torch.load(osp.join(self.processed_dir, f"data_{i}.pt"))
                data = transform(data)
                if save:
                    torch.save(data, osp.join(self.processed_dir, f"data_{i}.pt"))
            except Exception as e:
                print(f"Skipping {i} due to error {e}")

    def process(self, length=100):
        print("Processing data...")
        input("Press ENTER to continue")
        i = 0
        for file in tqdm(self.raw_file_names):
            data_load = json.load(open(osp.join(self.raw_dir, file)))
            right_landmarks = []
            left_landmarks = []
            landmarks = []
            for el in data_load:
                landmark_3d_raw = data_load[el]
                right_hand_landmarks = np.array(landmark_3d_raw['right_hand_3d'])
                left_hand_landmarks = np.array(landmark_3d_raw['left_hand_3d'])
                if right_hand_landmarks.shape[0] == 0:
                    right_hand_landmarks = np.ones((21, 3))*np.NaN
                if left_hand_landmarks.shape[0] == 0:
                    left_hand_landmarks = np.ones((21, 3))*np.NaN
                landmarks.append(np.concatenate((right_hand_landmarks, left_hand_landmarks), axis=0))
                right_landmarks.append(right_hand_landmarks)
                left_landmarks.append(left_hand_landmarks)
                metadata = data_load[el]['metadata']
                take_uid = metadata['take_uid']
                take_name = metadata['take_name']
                if len(landmarks) == length:
                    # data = Data(right_land =  torch.from_numpy(np.array(right_landmarks)),
                    #         left_land =  torch.from_numpy(np.array(left_landmarks)),
                    #         land = torch.from_numpy(np.array(landmarks)),
                    #         take_uid = take_uid,
                    #         take_name = take_name)
                    data = []
                    for l in range(length):
                        data.append(Data(land = torch.from_numpy(np.array(landmarks[l]))))
                    data, slices = self.collate(data)
                    data["take_uid"] = take_uid
                    data["take_name"] = take_name
                    torch.save(data, osp.join(self.processed_dir, f"data_{i}.pt"))
                    i += 1
                    right_landmarks = []
                    left_landmarks = []
                    landmarks = []
                
            # data = Data(right_land =  torch.from_numpy(np.array(right_landmarks)),
            #                 left_land =  torch.from_numpy(np.array(left_landmarks)),
            #                 land = torch.from_numpy(np.array(landmarks)),
            #                 take_uid = take_uid,
            #                 take_name = take_name)

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     print(f"Skipping {id} due to pre-filtering")
            #     continue
            
            # torch.save(data, osp.join(self.processed_dir, f"data_{i}.pt"))
            # i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data