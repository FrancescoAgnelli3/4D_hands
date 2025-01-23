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
        output = model(spline, times.to(device))
        loss = criterion(output, label.long())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (output.argmax(dim=1) == label).sum().item()
        batch_num += output.shape[0]

        scheduler.step(total_loss)
    train_epoch_loss = total_loss/len(train_loader)
    train_epoch_acc = total_acc/batch_num
    return train_epoch_loss, train_epoch_acc

def eval(model, times, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    batch_num = 0
    for batch in tqdm(test_loader):
        batch = tuple(b.to(device, dtype=torch.float) for b in batch)
        *valid_coeffs, action, label, obj, id = batch
        spline = controldiffeq.NaturalCubicSpline(times.to(device), valid_coeffs)
        output = model(spline, times.to(device))
        loss = criterion(output, label.long())
        total_loss += loss.item()
        total_acc += (output.argmax(dim=1) == label).sum().item()
        batch_num += output.shape[0]
        

    epoch_loss = total_loss/len(test_loader)
    epoch_acc = total_acc/batch_num
    return epoch_loss,  epoch_acc

