from tqdm import tqdm, trange
import torch_geometric.transforms as T
from utility import transforms as U
from utility.train_eval import train, eval
from dataloader import Assembly_dataloader
from torch.utils.data.dataloader import DataLoader
import torch
import os
from model.Make_model import make_model
from model.BasicCNN import BasicCNN
from models_handformer import hf_pose
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

torch.manual_seed(0)

current_directory = os.path.dirname(__file__)

# path = "/mnt/wwn-0x5000000000000001-part1/datasets/AssemblyHands/assembly101-download-scripts/data_our"
# path_loader = os.path.join(current_directory, "results/data/train_loader.pt")
path_loader = "/home/uniusr03/data/train_loader.pt"

batch_dim = 64
train_dataset = Assembly_dataloader(path_loader, train = True)
if os.path.exists(path_loader) is False:
    train_loader = train_dataset.save_data()
else:
    train_loader = train_dataset.load_data()
train_loader = DataLoader(train_loader, batch_size=batch_dim, shuffle=True)

# path_loader_val = os.path.join(current_directory, "results/data/val_loader.pt")
path_loader_val = "/home/uniusr03/data/test_loader.pt"

test_dataset = Assembly_dataloader(path_loader_val, train = False)
if os.path.exists(path_loader_val) is False:
    test_loader = test_dataset.save_data()
else:
    test_loader = test_dataset.load_data()
test_loader = DataLoader(test_loader, batch_size=batch_dim, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_times = torch.arange(100).float().to(device)
test_times = torch.arange(100).float().to(device)

for hc in [32]:  #4,8,16,32,64,128
    for model_name in ["NODE"]: #, "CNN" "NODE",
        print(f"Model: {model_name}, Hidden channels: {hc}")
        if model_name == "NODE":
                model = make_model(input_dim = 3, 
                    hid_dim = hc, 
                    hid_hid_dim = int(hc/2), 
                    num_layers = 2, 
                    num_nodes = 42, 
                    cheb_k = 2, 
                    embed_dim = 8, 
                    g_type = 'agc', 
                    output_dim = 24,  
                    solver = 'rk4') #'rk4', 'euler', 'dopri5'
                    # horizon = int(len(train_times[int(0.9*len(train_times)):])))
        if model_name == "CNN":
            model = BasicCNN(input_channels=3, output_channels=24, hid_channels=hc, kernel_size=3, stride=1, padding=1)
        model_name = model_name + "_" + str(hc)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        # Optional: Count trainable parameters only
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        model.to(device)
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        counter = 0
        train_loss = 1000
        # train
        model.train()
        for epoch in trange(30):
            old_train_loss = train_loss
            train_loss, train_acc = train(model, train_times, train_loader, optimizer, criterion, device, scheduler)
            with torch.no_grad():
                test_loss, test_acc = eval(model, test_times, test_loader, criterion, device)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
            # print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
            if train_loss < old_train_loss:
                counter = 0
            else:
                counter += 1
                if counter > 5:
                    break
        # save the test and train accuracy and loss in  json file in path
            
        with open(os.path.join(current_directory, f"results/accuracy_loss_{model_name}.json"), 'w') as f:
            json.dump({"train_loss": train_loss, 
                    "train_acc": train_acc, 
                    "test_loss": test_loss, 
                    "test_acc": test_acc, 
                    "epochs": epoch}, f, indent=4)

# print 