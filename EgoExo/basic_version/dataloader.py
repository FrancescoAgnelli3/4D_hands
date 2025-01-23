import torch
import numpy as np
from tqdm import tqdm
import os
import json
import os.path as osp
from torch.utils.data import Dataset, TensorDataset
import controldiffeq
from torch.utils.data.dataloader import DataLoader

class Ego4D_dataloader():
    def __init__(self, path, path_data = None, length = 100, noNaN = None):
        self.path = path
        self.path_data = path_data
        self.path_times = self.path.replace("loader", "times")
        self.length = length
        self.noNaN = noNaN
    
    def load_data(self):
        loader = torch.load(self.path)
        # times = torch.load(self.path_times)
        return loader#, times
    
    def save_data(self):
        print("Bulding the data...")
        dataset_landmarks = []
        dataset_task = []
        dataset_id = []
        for file in tqdm(os.listdir(self.path_data)):
            data = json.load(open(osp.join(self.path_data, file)))
            right_landmarks = []
            left_landmarks = []
            landmarks = []
            for el in data:
                landmark_3d_raw = data[el]
                right_hand_landmarks = np.array(landmark_3d_raw['right_hand_3d'])
                left_hand_landmarks = np.array(landmark_3d_raw['left_hand_3d'])
                if right_hand_landmarks.shape[0] == 0:
                    right_hand_landmarks = np.ones((21, 3))*np.NaN
                if left_hand_landmarks.shape[0] == 0:
                    left_hand_landmarks = np.ones((21, 3))*np.NaN
                hand_landmarks = np.concatenate([right_hand_landmarks, left_hand_landmarks], axis=0)
                landmarks.append(hand_landmarks)
                right_landmarks.append(right_hand_landmarks)
                left_landmarks.append(left_hand_landmarks)
                metadata = data[el]['metadata']
                take_uid = metadata['take_uid']
                task = metadata['take_name']
                task = get_task(task)
                if len(landmarks) == self.length:
                    landmarks = np.array(landmarks)
                    if self.noNaN is not None:
                        if np.isnan(landmarks).sum() < self.noNaN*(landmarks.size):
                            dataset_task.append(task)
                            landmarks = torch.tensor(landmarks)
                            dataset_landmarks.append(landmarks)
                            dataset_id.append(take_uid)
                    else:
                        dataset_task.append(task)
                        landmarks = torch.tensor(landmarks)
                        dataset_landmarks.append(landmarks)
                        dataset_id.append(take_uid)
                    landmarks = []
        print(len(dataset_landmarks))
        # dataset_land_padded, length = padding(dataset_landmarks)
        times = torch.arange(self.length).float()
        dataset_splineconv = SplineCoeff(torch.stack(dataset_landmarks), times)       
        data = TensorDataset(*dataset_splineconv, torch.tensor(dataset_task))
        # data = Ego4D(dataset_splineconv, dataset_task, dataset_id)
        torch.save(data, self.path)
        # torch.save(times, self.path_times)
        return data #, times

class Ego4D(Dataset):
    def __init__(self, data, labels, id, transform=None):
        self.data = data
        self.labels = labels
        self.id =  id
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        id = self.id[idx]
        if self.transform:
            data= self.transform(data)
        return data, label, id
                
def padding(dataset):
    print("Padding...")
    length = 0
    for data in dataset:
        length = np.max([length, len(data)])
    dataset_padded = []
    for data in dataset:
        N, M, T = data.shape
        padded_tensor = torch.full((length, M, T), float('nan'))
        padded_tensor[:N, :, :] = data
        dataset_padded.append(padded_tensor)
    return torch.stack(dataset_padded), length

def get_task(task):
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
    return task    

def SplineCoeff(dataset_landmarks, times):
    print("Calculating Spline Coeff...")
    dataset_splineconv = []
    land = dataset_landmarks.permute(0,2,1,3).float()
    dataset_splineconv = controldiffeq.natural_cubic_spline_coeffs(times, land)
    return dataset_splineconv

