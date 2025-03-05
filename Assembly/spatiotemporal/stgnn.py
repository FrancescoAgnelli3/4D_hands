import os
import glob
import torch.nn.functional as F
import networkx
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import Dataset

def normalize(y):
    max_norm = np.max(np.linalg.norm(y, axis=-1))
    return y/max_norm

def compute_velocity(points, delta_t=1):
	"""
	Compute the velocity among 3D points in a matrix.
	
	Args:
	points (np.ndarray): A numpy array of shape (N, 3) where N is the number of points and each point has 3 coordinates (x, y, z).
	
	Returns:
	np.ndarray: A numpy array of shape (N-1, 3) representing the velocity between consecutive points.
	"""
	velocities = (points[1:] - points[:-1])*delta_t
	norm_points = np.max(np.linalg.norm(points, axis=-1))
	return velocities/norm_points

def add_ohe(points):
	ohe = np.repeat(np.expand_dims(np.eye(points.shape[1]), axis=0), points.shape[0], axis=0)
	points = np.concatenate([points, ohe], axis=-1)
	return points

def load_files(folder_path, compute_velocity_b=True, ohe = False):

	# load train, val and test data	
	pattern = os.path.join(folder_path, '*rotate*.npy')
	file_names = glob.glob(pattern)
	data = []
	temp_index = [0]
	j = 0
	for file in file_names:
		d = np.load(file)
		# d = d.transpose(0,1,3,2)[:,:,0,:]
		d = d.transpose(0,1,3,2).reshape((d.shape[0], d.shape[1], d.shape[2]*d.shape[3]))
		d = np.transpose(d, (1, 2, 0)) # (t, n, 3)

		for i in range(d.shape[0]):
			d[i] = normalize(d[i])
		
		# add velocity
		if compute_velocity_b:
			d = np.pad(d, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=0)
			velocities = compute_velocity(d[:, :, :3]) # type: ignore
			d[1:, :, 3:] = velocities

		if ohe:
			d = add_ohe(d)
				
		distances = np.linalg.norm(np.diff(d, axis=0), axis=2)  # Shape (t-1, n)
		distances = np.sum(distances, axis=1)
		if np.max(distances)<5:
			data.append(d)
			temp_index.append(d.shape[0]+temp_index[-1])
		
		# j+=1
		# if j > 100:
		# 	break

	# data = np.array(data, dtype = np.float32)
	
	print("Data len: ", len(data))
	return data, temp_index

from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, Splitter
from typing import Union

path = "/home/studenti/agnelli/projects/4D_hands/Assembly/data"
data, temp_index = load_files(path, compute_velocity_b=True)
temp_index = temp_index
data = np.concatenate(data, axis=0)
edge_index = torch.tensor([[4,19],[3,16],[2,13],[1,10],[19,18],[16,15],[13,12],[10,9],[18,17],[15,14],[12,11],[9,8],[17,5],[14,5],[11,5],[8,5],[0,7],[7,6],[6,5],[20,5],[17,14],[14,11],[11,8]])
edge_index = torch.cat([edge_index,edge_index+21], dim=0).permute(1,0)
edge_weight = torch.ones(edge_index.shape[1])
adj = (edge_index, edge_weight)

torch_dataset = SpatioTemporalDataset(target=data, # type: ignore
                                      index=temp_index, # type: ignore
                                      connectivity=[adj],   # type: ignore                                    
                                      horizon=5,
                                      window=30,
                                      stride=1)
print(torch_dataset)

class AssemblySplitter(Splitter):
    def __init__(self,
                 val_lenght: Union[int, float] = 0.1,
                 test_lenght: Union[int, float] = 0.1):
        super(AssemblySplitter, self).__init__()
        self.val_lenght = val_lenght
        self.test_lenght = test_lenght

    def fit(self, dataset: SpatioTemporalDataset):

        test_len = int(self.test_lenght*len(dataset.index)) # type: ignore
        val_len = int(self.val_lenght*len(dataset.index)) # type: ignore
        tr_index_split = dataset.index[:int(len(dataset.index))-test_len-val_len] # type: ignore
        val_index_split = dataset.index[int(len(dataset.index))-test_len-val_len-1:int(len(dataset.index))-test_len] # type: ignore
        test_index_split = dataset.index[int(len(dataset.index))-test_len-1:] # type: ignore
        
        train_idxs = []
        if len(tr_index_split) > 1:
            for i in range(len(tr_index_split)-1):
                first_time, last_time = tr_index_split[i], tr_index_split[i+1]-dataset.horizon-dataset.window # type: ignore
                if (last_time - first_time) > 0:
                    indices_after = first_time <= dataset.indices
                    indices_before = dataset.indices < last_time
                    indices = np.nonzero(indices_after & indices_before).ravel()
                    train_idxs.append(indices)
            train_idxs = np.concatenate(train_idxs).ravel()
        
        val_idxs = []
        if len(val_index_split) > 1:
            for i in range(len(val_index_split)-1):
                first_time, last_time = val_index_split[i], val_index_split[i+1]-dataset.horizon-dataset.window # type: ignore
                if (last_time - first_time) > 0:
                    indices_after = first_time <= dataset.indices
                    indices_before = dataset.indices < last_time
                    indices = np.nonzero(indices_after & indices_before).ravel()
                    val_idxs.append(indices)
            val_idxs = np.concatenate(val_idxs).ravel()

        test_idxs = []
        if len(test_index_split) > 1:
            for i in range(len(test_index_split)-1):
                first_time, last_time = test_index_split[i], test_index_split[i+1]-dataset.horizon-dataset.window # type: ignore
                if (last_time - first_time) > 0:
                    indices_after = first_time <= dataset.indices
                    indices_before = dataset.indices < last_time
                    indices = np.nonzero(indices_after & indices_before).ravel()
                    test_idxs.append(indices)
            test_idxs = np.concatenate(test_idxs).ravel()
        
        self.set_indices(train_idxs, val_idxs, test_idxs)

# Normalize data using mean and std computed over time and node dimensions
splitter = AssemblySplitter(val_lenght=0.0, test_lenght=0.2)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    # scalers=scalers,
    splitter=splitter,
    batch_size=1024,
)

dm.setup()
print(dm)

import torch.nn as nn
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, GraphConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation
from torch.utils.tensorboard import SummaryWriter

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

class TimeSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 t_layers: int = 1,
                 gnn_layers: int = 2,
                 att: bool = True):
        super(TimeSpaceModel, self).__init__()

        self.space = False
        if gnn_layers > 0:
            self.space = True
        self.att = att
        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 n_layers=2,
                                 cell='gru',
                                 return_only_last_state=True)

        if self.space:
            self.space_nn = nn.ModuleList()
            for i in range(gnn_layers):
                self.space_nn.append(GraphConv(input_size=hidden_size, output_size=hidden_size, root_weight=False))
            self.skip_connection = nn.Linear(hidden_size, hidden_size)
            
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        # split left and right hand 
        h = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(h)  # temporal processing
        if self.space:
            h_skip = self.skip_connection(h)
            for i in range(len(self.space_nn)):
                h = self.space_nn[i](h, edge_index)  # spatial processing
            h = h + h_skip
        x_out = self.decoder(h)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

def train_test(model, train_loader, test_loader, criterion, optimizer, metric, device, epochs=10, scheduler=None, log_dir="runs"):
    model.to(device)
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_rec = 0.0
        running_len = 0.0
        running_metric = 0.0
        total_samples = 0
        
        for batch in train_loader:
            inputs, targets = batch.input.to(device), batch.target.to(device)
            x = inputs.x
            edge_index = inputs.edge_index.to(dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(x, edge_index)
            diff_pred = torch.norm(outputs[:,:,edge_index[0], :3]-outputs[:,:,edge_index[1], :3], dim=-1)
            diff_true = torch.norm(targets.y[:,:,edge_index[0], :3]-targets.y[:,:,edge_index[1], :3], dim=-1)
            rec_error = metric(outputs, targets.y)
            len_error = criterion(diff_pred, diff_true)
            loss = rec_error + len_error
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            running_rec += rec_error.item() * x.size(0)
            running_len += len_error.item() * x.size(0)
            running_metric += criterion(outputs, targets.y).item() * x.size(0)
            total_samples += x.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_rec = running_rec / total_samples
        epoch_len = running_len / total_samples
        epoch_metric = running_metric / total_samples

        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Metric/Train", epoch_metric, epoch)
        writer.add_scalar("Rec_Error/Train", epoch_rec, epoch)
        writer.add_scalar("Len_Error/Train", epoch_len, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        
        if scheduler is not None:
            scheduler.step(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Metric: {epoch_metric:.4f}, Train rec_error: {epoch_rec:.4f}, Train len_error: {epoch_len:.4f}") 
        print(f"learning_rate: {optimizer.param_groups[0]['lr']}")
        
        model.eval()
        running_loss = 0.0
        running_metric = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch.input.to(device), batch.target.to(device)
                x = inputs.x
                edge_index = inputs.edge_index.to(dtype=torch.long)
                
                outputs = model(x, edge_index)
                loss = metric(outputs, targets.y)
                
                running_loss += loss.item() * x.size(0)
                running_metric += criterion(outputs, targets.y).item() * x.size(0)
                total_samples += x.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_metric = running_metric / total_samples

        writer.add_scalar("Loss/Test", epoch_loss, epoch)
        writer.add_scalar("Metric/Test", epoch_metric, epoch)
        
        print(f"Test Loss: {epoch_loss:.4f}, Test Metric: {epoch_metric:.4f}")
    
    writer.close()

class TimeWristHandModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 t_layers: int = 1,
                 gnn_layers: int = 2):
        super(TimeWristHandModel, self).__init__()

        self.space = False
        if gnn_layers > 0:
            self.space = True
        self.num_nodes = n_nodes
        self.encoder_hands_left = nn.Linear(input_size, hidden_size)
        self.encoder_hands_right = nn.Linear(input_size, hidden_size)
        self.encoder_wrist_left = nn.Linear(input_size, hidden_size)
        self.encoder_wrist_right = nn.Linear(input_size, hidden_size)

        self.node_embeddings_hands = NodeEmbedding(int(n_nodes/2), hidden_size)
        self.node_embeddings_wrist = NodeEmbedding(2, hidden_size)

        # self.time_nn_hand = RNN(input_size=hidden_size,
        #             hidden_size=hidden_size,
        #             n_layers=t_layers,
        #             cell='gru',
        #             return_only_last_state=True)
        self.time_nn_hand_left = RNN(input_size=hidden_size,
                    hidden_size=hidden_size,
                    n_layers=t_layers,
                    cell='gru',
                    return_only_last_state=True)
        self.time_nn_hand_right = RNN(input_size=hidden_size,
                    hidden_size=hidden_size,
                    n_layers=t_layers,
                    cell='gru',
                    return_only_last_state=True)
        self.time_nn_wrist_left = RNN(input_size=hidden_size,
                    hidden_size=hidden_size,
                    n_layers=t_layers,
                    cell='gru',
                    return_only_last_state=True)
        self.time_nn_wrist_right = RNN(input_size=hidden_size,
                                       hidden_size=hidden_size,
                                       n_layers=t_layers,
                                       cell='gru',
                                       return_only_last_state=True)
        
        if self.space:
            self.space_nn_left = nn.ModuleList()
            self.space_nn_right = nn.ModuleList()
            for i in range(gnn_layers):
                self.space_nn_left.append(GraphConv(input_size=hidden_size, output_size=hidden_size))
                self.space_nn_right.append(GraphConv(input_size=hidden_size, output_size=hidden_size))

        self.decoder_hands_left = nn.Linear(hidden_size, input_size * horizon)
        self.decoder_hands_right = nn.Linear(hidden_size, input_size * horizon)
        self.decoder_wrist_left = nn.Linear(hidden_size, input_size * horizon)
        self.decoder_wrist_right = nn.Linear(hidden_size, input_size * horizon)

        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index):
        # x: [batch time nodes features]
        # x_wrist = x[:,:,[5,26],:]
        # x_left = x[:,:,:int(x.shape[2]/2),:] - x_wrist[:,:,0,:].unsqueeze(2)
        # x_right = x[:,:,int(x.shape[2]/2):,:] - x_wrist[:,:,1,:].unsqueeze(2)
        # x_hand = torch.cat((x_left, x_right), dim=2)
        # x_enc_hand = self.encoder_hands(x_hand)  # linear encoder: x_enc = xΘ + b
        # x_enc_wrist = self.encoder_wrist(x_wrist)
        # x_emb_hand = x_enc_hand + self.node_embeddings_hands()  # add node-identifier embeddings
        # x_emb_wrist = x_enc_wrist + self.node_embeddings_wrist()  # add node-identifier embeddings
        # h_hand = self.time_nn_hand(x_emb_hand)  # temporal processing: x=[b t n f] -> h=[b n f]
        # h_wrist = self.time_nn_wrist(x_emb_wrist)  # temporal processing: x=[b t n f] -> h=[b n f]
        # if self.space:
        #     for i in range(len(self.space_nn)):
        #         h_hand = self.space_nn[i](h_hand, edge_index)  # spatial processing
        # x_out_wrist = self.decoder_hands(h_wrist)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        # x_out_hand = self.decoder_hands(h_hand)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        # x_horizon_wrist = self.rearrange(x_out_wrist)
        # x_horizon_hand = self.rearrange(x_out_hand)
        # return x_horizon_hand, x_horizon_wrist
        x_wrist = x[:,:,[5,26],:]
        x_hand_left = x[:,:,:int(x.shape[2]/2),:] - x_wrist[:,:,0,:].unsqueeze(2)
        x_hand_right = x[:,:,int(x.shape[2]/2):,:] - x_wrist[:,:,1,:].unsqueeze(2)
        x_enc_hand_left = self.encoder_hands_left(x_hand_left)  # linear encoder: x_enc = xΘ + b
        x_enc_hand_right = self.encoder_hands_right(x_hand_right)  # linear encoder: x_enc = xΘ + b
        x_enc_wrist_left = self.encoder_wrist_left(x_wrist[:,:,0,:].unsqueeze(2))
        x_enc_wrist_right = self.encoder_wrist_right(x_wrist[:,:,1,:].unsqueeze(2))
        x_emb_hand_left = x_enc_hand_left + self.node_embeddings_hands()  # add node-identifier embeddings
        x_emb_hand_right = x_enc_hand_right + self.node_embeddings_hands()  # add node-identifier embeddings
        h_hand_left = self.time_nn_hand_left(x_emb_hand_left)  # temporal processing: x=[b t n f] -> h=[b n f]
        h_hand_right = self.time_nn_hand_right(x_emb_hand_right)  # temporal processing: x=[b t n f] -> h=[b n f]
        h_wrist_left = self.time_nn_wrist_left(x_enc_wrist_left)  # temporal processing: x=[b t n f] -> h=[b n f]
        h_wrist_right = self.time_nn_wrist_right(x_enc_wrist_right)  # temporal processing: x=[b t n f] -> h=[b n f]
        if self.space:
            for i in range(len(self.space_nn_left)):
                h_hand_left = self.space_nn_left[i](h_hand_left, edge_index[:,:int(edge_index.shape[-1]/2)])  # spatial processing
                h_hand_right = self.space_nn_right[i](h_hand_right, edge_index[:,int(edge_index.shape[-1]/2):]-21)  # spatial processing
        x_out_wrist_left = self.decoder_wrist_left(h_wrist_left)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_out_wrist_right = self.decoder_wrist_right(h_wrist_right)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_out_hand_left = self.decoder_hands_left(h_hand_left)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_out_hand_right = self.decoder_hands_right(h_hand_right)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_out_hand = torch.cat((x_out_hand_left, x_out_hand_right), dim=1)
        x_out_wrist = torch.cat((x_out_wrist_left, x_out_wrist_right), dim=1)
        x_horizon_wrist = self.rearrange(x_out_wrist)
        x_horizon_hand = self.rearrange(x_out_hand)
        return x_horizon_hand, x_horizon_wrist

def train_test_wrist(model, train_loader, test_loader, criterion, optimizer, metric, device, epochs=10, scheduler=None, log_dir="runs"):
    model.to(device)
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_metric = 0.0
        running_rec = 0.0
        running_rec_wrist = 0.0
        running_len = 0.0
        total_samples = 0
        
        for batch in train_loader:
            inputs, targets = batch.input.to(device), batch.target.to(device)
            x = inputs.x
            edge_index = inputs.edge_index.to(dtype=torch.long)

            optimizer.zero_grad()
            outputs, outputs_wrist = model(x, edge_index)
            true_wrist = targets.y[:,:,[5,26],:]
            true_left = targets.y[:,:,:int(targets.y.shape[2]/2),:] - true_wrist[:,:,0,:].unsqueeze(2)
            true_right = targets.y[:,:,int(targets.y.shape[2]/2):,:] - true_wrist[:,:,1,:].unsqueeze(2)
            true_hand = torch.cat((true_left, true_right), dim=2)
            diff_pred = torch.norm(outputs[:,:,edge_index[0], :3]-outputs[:,:,edge_index[1], :3], dim=-1)
            diff_true = torch.norm(true_hand[:,:,edge_index[0], :3]-true_hand[:,:,edge_index[1], :3], dim=-1)
            
            l_1, l_2, l_3 = 1, 1, 1
            error_rec = metric(outputs, true_hand)
            error_rec_wrist = metric(outputs_wrist, true_wrist)
            error_len = criterion(diff_pred, diff_true)
            loss = l_1 * error_rec + l_2 * error_rec_wrist + l_3 * error_len
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            running_rec += error_rec.item() * x.size(0)
            running_rec_wrist += error_rec_wrist.item() * x.size(0)
            running_len += error_len.item() * x.size(0)
            hand_left = outputs_wrist[:,:,0,:].unsqueeze(2) + outputs[:,:,:21,:]
            hand_right = outputs_wrist[:,:,1,:].unsqueeze(2) + outputs[:,:,21:,:]
            hands = torch.cat((hand_left, hand_right), dim=2)
            running_metric += metric(hands, targets.y).item() * x.size(0)
            total_samples += x.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_rec = running_rec / total_samples
        epoch_rec_wrist = running_rec_wrist / total_samples
        epoch_len = running_len / total_samples
        epoch_metric = running_metric / total_samples

        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Metric/Train", epoch_metric, epoch)
        writer.add_scalar("Rec/Train", epoch_rec, epoch)
        writer.add_scalar("Rec_Wrist/Train", epoch_rec_wrist, epoch)
        writer.add_scalar("Len/Train", epoch_len, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Metric: {epoch_metric:.4f}, Train Rec: {epoch_rec:.4f}, Train Rec Wrist: {epoch_rec_wrist:.4f}, Train Len: {epoch_len:.4f}")
        print(f"learning_rate: {optimizer.param_groups[0]['lr']}")

        if scheduler is not None:
            scheduler.step(epoch_loss)
        
        model.eval()
        running_loss = 0.0
        running_metric = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch.input.to(device), batch.target.to(device)
                x = inputs.x
                edge_index = inputs.edge_index.to(dtype=torch.long)
                
                outputs, outputs_wrist = model(x, edge_index)
                true_wrist = targets.y[:,:,[5,26],:]
                true_left = targets.y[:,:,:int(targets.y.shape[2]/2),:] - true_wrist[:,:,0,:].unsqueeze(2)
                true_right = targets.y[:,:,int(targets.y.shape[2]/2):,:] - true_wrist[:,:,1,:].unsqueeze(2)
                true_hand = torch.cat((true_left, true_right), dim=2)
                diff_pred = torch.norm(outputs[:,:,edge_index[0], :3]-outputs[:,:,edge_index[1], :3], dim=-1)
                diff_true = torch.norm(true_hand[:,:,edge_index[0], :3]-true_hand[:,:,edge_index[1], :3], dim=-1)
                loss = criterion(outputs, true_hand) + criterion(outputs_wrist, true_wrist) + criterion(diff_pred, diff_true)
                
                running_loss += loss.item() * x.size(0)
                hand_left = outputs_wrist[:,:,0,:].unsqueeze(2) + outputs[:,:,:21,:]
                hand_right = outputs_wrist[:,:,1,:].unsqueeze(2) + outputs[:,:,21:,:]
                hands = torch.cat((hand_left, hand_right), dim=2)
                running_metric += metric(hands, targets.y).item() * x.size(0)
                total_samples += x.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_metric = running_metric / total_samples

        writer.add_scalar("Loss/Test", epoch_loss, epoch)
        writer.add_scalar("Metric/Test", epoch_metric, epoch)
        
        print(f"Test Loss: {epoch_loss:.4f}, Test Metric: {epoch_metric:.4f}")
    
    writer.close()

hidden_size = 64   #@param
t_layers = 2     #@param
gnn_layers = 0      #@param

input_size = torch_dataset.n_channels   # 1 channel
n_nodes = torch_dataset.n_nodes         # 207 nodes
horizon = torch_dataset.horizon         # 12 time steps

stgnn = TimeWristHandModel(input_size=input_size,
                           n_nodes=n_nodes,
                           horizon=horizon,
                           hidden_size=hidden_size,
                           t_layers=t_layers,
                           gnn_layers=gnn_layers)
model_name = "stgnn_wrist_hand"+"_hc_"+str(hidden_size)+"_tl_"+str(t_layers)+"_gl_"+str(gnn_layers)

print_model_size(stgnn)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
epochs = 60

# Data loaders
train_loader = dm.train_dataloader()
test_loader = dm.test_dataloader()

criterion = nn.MSELoss()
metric = nn.L1Loss()
optimizer = torch.optim.Adam(stgnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

log_dir = "/home/studenti/agnelli/projects/4D_hands/Assembly/spatiotemporal/runs/" + model_name

# Train and test the model
train_test_wrist(stgnn, train_loader, test_loader, criterion, optimizer, metric, device, epochs, scheduler, log_dir= log_dir)

torch.save(stgnn.state_dict(), "/home/studenti/agnelli/projects/4D_hands/Assembly/spatiotemporal/out/"+model_name+".pt")