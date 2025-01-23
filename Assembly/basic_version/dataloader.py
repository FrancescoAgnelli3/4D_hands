import torch
import numpy as np
from tqdm import tqdm
import os
import json
import os.path as osp
from torch.utils.data import Dataset, TensorDataset
import controldiffeq
from torch.utils.data.dataloader import DataLoader

class Assembly_dataloader():
    def __init__(self, path, path_data = None, length = 100, train = True):
        self.path = path
        self.path_data = path_data
        self.path_times = self.path.replace("loader", "times")
        self.length = length
        self.train = train
    
    def load_data(self):
        print("Loading the data...")
        loader = torch.load(self.path)
        # times = torch.load(self.path_times)
        return loader#, times
    
    def save_data(self):
        print("Bulding the data...")
        dataset_landmarks = []
        dataset_action = []
        dataset_verb = []
        dataset_obj = []
        dataset_id = []
        for file in tqdm(os.listdir(self.path_data)):
            if self.train:
                string = 'tr'
            else:
                string = 'te'
            if file.startswith(string):
                landmarks = np.load(osp.join(self.path_data, file)) #3xTx21X2
                landmarks = landmarks.transpose(1, 0, 3, 2) #Tx3x21x2
                landmarks = landmarks.reshape(landmarks.shape[0], landmarks.shape[1], -1) #Tx3XJ
                max_land = np.max(landmarks, keepdims=True)
                min_land = np.min(landmarks, keepdims=True)
                landmarks = (landmarks-min_land)/(max_land-min_land)
                # get the reminder of the division between len(landmarks) and self.length
                if len(landmarks) > self.length:
                    rem = len(landmarks) % self.length
                    landmarks = landmarks[:len(landmarks)-rem].transpose(0,2,1) #TxJx3
                    landmarks = landmarks.reshape(int(landmarks.shape[0]/self.length), self.length, landmarks.shape[1], -1) #LxTxJx3
                    # transform landmarks of dim BxTxDXJ in a list of B vectors of dimension TxDXJ
                    landmarks = [torch.tensor(landmarks[i]) for i in range(landmarks.shape[0])]
                    dataset_landmarks.append(landmarks)
                    for repeat in range(len(landmarks)):
                        dataset_id.append(int(file.split("_")[0].replace(string, "")))
                        dataset_action.append(int(file.split("_")[1].replace("a", "")))
                        dataset_verb.append(int(file.split("_")[2].replace("v", "")))
                        dataset_obj.append(int(file.split("_")[3].replace("n", "")))
        dataset_landmarks = [item for sublist in dataset_landmarks for item in sublist]

        # dataset_land_padded, length = padding(dataset_landmarks)
        times = torch.arange(self.length).float()
        dataset_splineconv = SplineCoeff(torch.stack(dataset_landmarks), times)   
        data = TensorDataset(*dataset_splineconv, torch.tensor(dataset_action), torch.tensor(dataset_verb), torch.tensor(dataset_obj), torch.tensor(dataset_id))
        # data = Ego4D(dataset_splineconv, dataset_task, dataset_id)
        torch.save(data, self.path)
        # torch.save(times, self.path_times)
        return data #, times

def SplineCoeff(dataset_landmarks, times):
    print("Calculating Spline Coeff...")
    dataset_splineconv = []
    land = dataset_landmarks.permute(0,2,1,3).float()
    dataset_splineconv = controldiffeq.natural_cubic_spline_coeffs(times, land)
    return dataset_splineconv

