import os
import glob
import torch
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

# dnri
import dnri.training.train as train
from dnri.utils.data_utils import *

from tsl.data import SpatioTemporalDataset
from tsl.ops.connectivity import edge_index_to_adj
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation


def normalize(y):
	m1 = y.mean(axis=(1,0))
	s1 = y.std(axis=(1,0))
	return (y-m1)/s1

# def normalize(y):
#     return y

def equalize_ndarray(arr, axis=0, target_size=None):
	"""
	Equalize a numpy ndarray along a specified axis by undersampling or uppersampling by a fixed quantity.
	
	Parameters:
	arr (numpy.ndarray): Input array to be equalized.
	axis (int): Axis along which to equalize the array.
	target_size (int): Target size for the specified axis. If None, the size of the smallest dimension along the axis is used.
	
	Returns:
	numpy.ndarray: Equalized array.
	"""
      
	if target_size is None:
		target_size = min(arr.shape[axis])
	
	current_size = arr.shape[axis]
	indices = np.arange(current_size)
      
	delta_t = target_size/current_size
	
	if current_size > target_size:
		# Undersampling
		indices = np.linspace(0, current_size - 1, target_size).astype(int)
	elif current_size < target_size:
		# Oversampling
		indices = np.linspace(0, current_size - 1, target_size).astype(int)
	
	equalized_arr = np.take(arr, indices, axis=axis)
	return equalized_arr, delta_t

def compute_velocity(points, delta_t=1):
	"""
	Compute the velocity among 3D points in a matrix.
	
	Args:
	points (np.ndarray): A numpy array of shape (N, 3) where N is the number of points and each point has 3 coordinates (x, y, z).
	
	Returns:
	np.ndarray: A numpy array of shape (N-1, 3) representing the velocity between consecutive points.
	"""
	velocities = (points[1:] - points[:-1])*delta_t
	return velocities

def add_ohe(points):
	ohe = np.repeat(np.expand_dims(np.eye(points.shape[1]), axis=0), points.shape[0], axis=0)
	points = np.concatenate([points, ohe], axis=-1)
	return points

def load_files(folder_path, compute_velocity_b=True, equalize=True, ohe = True, target_size=100):

	# load train, val and test data	
	pattern = os.path.join(folder_path, 'tr*' + '*.npy')
	file_names = glob.glob(pattern)
	data = []
	for file in file_names:
		d = np.load(file)
		# d = d.transpose(0,1,3,2)[:,:,0,:]
		d = d.transpose(0,1,3,2).reshape((d.shape[0], d.shape[1], d.shape[2]*d.shape[3]))
		d = np.transpose(d, (1, 2, 0)) # (t, n, 3)

		if equalize:
			d, delta_t = equalize_ndarray(d, axis=0, target_size=target_size)
		
		
		# add velocity
		if compute_velocity_b:
			d = np.pad(d, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=0)
			velocities = compute_velocity(d[:, :, :3], delta_t)
			d[1:, :, 3:] = velocities
		d = normalize(d)

		if ohe:
			d = add_ohe(d)	
		data.append(d)

	data = np.array(data, dtype = np.float32)

	print("Data len: ", len(data))
	return data

path = "/home/studenti/agnelli/projects/nri/data"
data = load_files(path)
data = data.reshape(-1, 42, 48)
edge_index = torch.tensor([[4,19],[3,16],[2,13],[1,10],[19,18],[16,15],[13,12],[10,9],[18,17],[15,14],[12,11],[9,8],[17,5],[14,5],[11,5],[8,5],[0,7],[7,6],[6,5],[20,5]])
edge_index = torch.cat([edge_index,edge_index+21], dim=0).permute(1,0)
edge_weight = torch.ones(edge_index.shape[1])
adj = (edge_index, edge_weight)

torch_dataset = SpatioTemporalDataset(target=data,
                                      connectivity=adj,
                                      horizon=20,
                                      window=80,
                                      stride=100)
print(torch_dataset)


# Normalize data using mean and std computed over time and node dimensions
scalers = {'target': StandardScaler(axis=(0, 1))}
splitter = TemporalSplitter(val_len=0.0, test_len=0.1)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    # scalers=scalers,
    splitter=splitter,
    batch_size=64,
)

dm.setup()
print(dm)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight = None):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

def train_test(model, train_loader, test_loader, criterion, optimizer, metric, device, epochs=10, scheduler=None, log_dir="runs"):
    model.to(device)
    writer = SummaryWriter(log_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_metric = 0.0
        total_samples = 0
        
        for batch in train_loader:
            inputs, targets = batch.input.to(device), batch.target.to(device)
            x = inputs.x
            # edge_index = dense_to_sparse(inputs.edge_index.to(dtype=torch.long))[0]
            edge_index = inputs.edge_index.to(dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(x, edge_index)
            diff_pred = torch.norm(outputs[:,:,edge_index[0], :3]-outputs[:,:,edge_index[1], :3], dim = -1)
            diff_true = torch.norm(targets.y[:,:,edge_index[0], :3]-targets.y[:,:,edge_index[1], :3], dim = -1)
            loss = criterion(outputs, targets.y) + criterion(diff_pred, diff_true)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            running_metric += metric(outputs, targets.y).item() * x.size(0)
            total_samples += x.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_metric = running_metric / total_samples

        if scheduler is not None:
            scheduler.step(epoch_train_loss)

        writer.add_scalar("Loss", epoch_train_loss, epoch)
        writer.add_scalar("Metric", epoch_train_metric, epoch)

        running_loss = 0.0
        running_metric = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch.input.to(device), batch.target.to(device)
                x = inputs.x
                # edge_index = dense_to_sparse(inputs.edge_index.to(dtype=torch.long))[0]
                edge_index = inputs.edge_index.to(dtype=torch.long)

                outputs = model(x, edge_index)
                loss = criterion(outputs, targets.y)
                
                running_loss += loss.item() * x.size(0)
                running_metric += metric(outputs, targets.y).item() * x.size(0)
                total_samples += x.size(0)

        epoch_test_loss = running_loss / total_samples
        epoch_test_metric = running_metric / total_samples

        writer.add_scalar("Loss/test", epoch_test_loss, epoch)
        writer.add_scalar("Metric/test", epoch_test_metric, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Train Metric: {epoch_train_metric:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Metric: {epoch_test_metric:.4f}")
    
    writer.close()

for hc in [16, 32, 64, 128, 256, 512]:
    hidden_size = hc   #@param
    rnn_layers = 3     #@param
    gnn_kernel = 3     #@param
    model_name = f"hc_{hidden_size}_rnn_{rnn_layers}_gnn_{gnn_kernel}"

    input_size = torch_dataset.n_channels   # 1 channel
    n_nodes = torch_dataset.n_nodes         # 207 nodes
    horizon = torch_dataset.horizon         # 12 time steps

    stgnn = TimeThenSpaceModel(input_size=input_size,
                            n_nodes=n_nodes,
                            horizon=horizon,
                            hidden_size=hidden_size,
                            rnn_layers=rnn_layers,
                            gnn_kernel=gnn_kernel)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    learning_rate = 0.001
    epochs = 100

    # Data loaders
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    criterion = nn.MSELoss()
    metric = nn.L1Loss()
    optimizer = optim.Adam(stgnn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train and test the model
    train_test(stgnn, train_loader, test_loader, criterion, optimizer, metric, device, epochs, scheduler, log_dir=f"runs/{model_name}")
