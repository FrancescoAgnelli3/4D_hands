import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import tqdm
import random
from tqdm import tqdm

def train(model, tr_loader, optimizer, criterion, device, scheduler):
    model.train()

    prev_h = None
    total_loss = 0
    total_acc = 0
    num_batch = 0
    for el in tqdm(tr_loader):
        # Reset gradients from previous step
        model.zero_grad()
        loss = 0
        land = el.land.reshape(-1, 100, 42, 3)
        batch = torch.arange(land.shape[0]).repeat_interleave(42)
        batch = batch.to(device)
        num_batch += land.shape[0]
        for i in range(land.shape[1]):
            snapshot = land[:,i,:,:].to(device).to(torch.float32) 
            snapshot = snapshot.reshape(-1, 3)  
            
            # Perform a forward pass
            y, h = model.forward(snapshot, batch, prev_h)
            prev_h = h
            target = el.task
            loss += criterion(y, target.to(device))


        # Perform a backward pass to calculate the gradients
        loss.backward()
        total_loss += loss.item()
        total_acc += (y.cpu().argmax(1) == target).sum().item()

        # Update parameters
        optimizer.step()

        # if you don't detatch previous state you will get backprop error
        if prev_h is not None:
            prev_h = prev_h.detach()
    scheduler.step(total_loss/len(tr_loader))
        
    return total_loss/len(tr_loader), total_acc/num_batch


def eval(model, loader, criterion, device):
    with torch.no_grad():
        prev_h = None
        total_loss = 0
        total_acc = 0
        num_batch = 0
        for el in loader:
        # Reset gradients from previous step
            model.zero_grad()
            loss = 0
            land = el.land.reshape(-1, 100, 42, 3)
            batch = torch.arange(land.shape[0]).repeat_interleave(42)
            batch = batch.to(device)
            num_batch += land.shape[0]
            for i in range(land.shape[1]):
                snapshot = land[:,i,:,:].to(device).to(torch.float32) 
                snapshot = snapshot.reshape(-1, 3)    
            
                # Perform a forward pass
                y, h = model.forward(snapshot, batch, prev_h)
                prev_h = h
                target = el.task
                loss += criterion(y, target.to(device))

                total_loss += loss.item()
                total_acc += (y.cpu().argmax(1) == target).sum().item()

            # if you don't detatch previous state you will get backprop error
            if prev_h is not None:
                prev_h = prev_h.detach()
        
    return total_loss/len(loader), total_acc/num_batch
        