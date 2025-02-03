import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import controldiffeq

def train(model, times, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0
    total_acc = 0
    batch_num = 0
    for batch in tqdm(train_loader):
        batch = tuple(b.to(device, dtype=torch.float) for b in batch)
        *valid_coeffs, action, label, obj, id = batch
        spline = controldiffeq.NaturalCubicSpline(times.to(device), valid_coeffs)
        optimizer.zero_grad()
        output = model(spline, times[:int(0.9*len(times))].to(device))
        time_target = times[int(0.9*len(times)):].to(device)
        target = []
        for i in time_target:
            target.append(spline.evaluate(i)) 
        target = torch.stack(target).permute(1,0,2,3) 
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_num += output.shape[0]

        scheduler.step(total_loss)
    train_epoch_loss = total_loss/len(train_loader)
    return train_epoch_loss, None

def eval(model, times, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    batch_num = 0
    for batch in test_loader:
        batch = tuple(b.to(device, dtype=torch.float) for b in batch)
        *valid_coeffs, label = batch
        spline = controldiffeq.NaturalCubicSpline(times.to(device), valid_coeffs)
        output = model(spline, times[:int(0.9*len(times))].to(device))
        time_target = times[int(0.9*len(times)):].to(device)
        target = []
        for i in time_target:
            target.append(spline.evaluate(i))
        target = torch.stack(target).permute(1,0,2,3) 
        loss = criterion(output, target)
        total_loss += loss.item()
        batch_num += output.shape[0]
        
    epoch_loss = total_loss/len(test_loader)
    return epoch_loss, None

