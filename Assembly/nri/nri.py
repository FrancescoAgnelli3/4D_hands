import os
import glob
import torch.nn.functional as F
import networkx
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import Dataset

# dnri
from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
import dnri.training.train_utils as train_utils
import dnri.training.train as train
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc
from dnri.utils.data_utils import *

def normalize(y):
	m1 = y.mean(axis=(1,0))
	s1 = y.std(axis=(1,0))
	return (y-m1)/s1

# def normalize(y):
#     return y

# def normalize(y):
#     left_wr = y[5,:]
#     y = y - left_wr
#     y = rotate_and_normalize(y)
#     return y

# def rotation_matrix_from_vectors(vec1, vec2):
#     """Find the rotation matrix that aligns vec1 to vec2."""
#     a = vec1 / np.linalg.norm(vec1)
#     b = vec2 / np.linalg.norm(vec2)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
    
#     if s == 0:
#         return np.eye(3)  # Already aligned
    
#     vx = np.array([[0, -v[2], v[1]],
#                    [v[2], 0, -v[0]],
#                    [-v[1], v[0], 0]])
#     R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
#     return R

# def rotate_and_normalize(y):
#     """Rotate and normalize the vectors so that y[20] aligns with (1,0,0)."""
#     target = np.array([1, 0, 0])
#     R = rotation_matrix_from_vectors(y[20,:], target)

#     y = y / np.linalg.norm(y[20,:])  # Normalize  
#     y = (R @ y.T).T  # Apply rotation
#     return y
   
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

def load_files(folder_path, compute_velocity_b=False, substring='*.npy', equalize=False, eq_axis=0, target_size=150):

	# load train, val and test data	
	pattern = os.path.join(folder_path, 'tr*' + substring)
	file_names = glob.glob(pattern)
	train_data = []
	for file in file_names:
		d = np.load(file)
		# d = d.transpose(0,1,3,2)[:,:,0,:]
		d = d.transpose(0,1,3,2).reshape((d.shape[0], d.shape[1], d.shape[2]*d.shape[3]))
		d = np.transpose(d, (1, 2, 0))

		if equalize:
			d, delta_t = equalize_ndarray(d, axis=eq_axis, target_size=target_size)
		
		# add velocity
		if compute_velocity_b:
			d = np.pad(d, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=0)
			velocities = compute_velocity(d[:, :, :3], delta_t)
			d[1:, :, 3:] = velocities
      
		for i in range(d.shape[0]):
			d[i] = normalize(d[i])

		if np.max(np.sum((np.linalg.norm(np.diff(d, axis=0), axis=2)), axis = 1)) < 25:
			train_data.append(d)
	
	pattern = os.path.join(folder_path, 'va*'+substring)
	file_names = glob.glob(pattern)
	val_data = []
	for file in file_names:
		#print("Loading file:", file)
		d = np.load(file)
		d = d.transpose(0,1,3,2).reshape((d.shape[0], d.shape[1], d.shape[2]*d.shape[3]))
		d = np.transpose(d, (1, 2, 0))
		if equalize:
			d, delta_t = equalize_ndarray(d, axis=eq_axis, target_size=target_size)

		# add velocity
		if compute_velocity_b:
			d = np.pad(d, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=0)
			velocities = compute_velocity(d[:, :, :3], delta_t)
			d[1:, :, 3:] = velocities
   	
		for i in range(d.shape[0]):
			d[i] = normalize(d[i])
		
		if np.max(np.sum((np.linalg.norm(np.diff(d, axis=0), axis=2)), axis = 1)) < 25:
			val_data.append(d)
	
	pattern = os.path.join(folder_path, 'te*'+substring)
	file_names = glob.glob(pattern)
	test_data = []
	for file in file_names:
		#print("Loading file:", file)
		d = np.load(file)
		d = d.transpose(0,1,3,2).reshape((d.shape[0], d.shape[1], d.shape[2]*d.shape[3]))
		d = np.transpose(d, (1, 2, 0))
		if equalize:
			d, delta_t = equalize_ndarray(d, axis=eq_axis, target_size=target_size)

		# add velocity
		if compute_velocity_b:
			d = np.pad(d, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=0)
			velocities = compute_velocity(d[:, :, :3], delta_t)
			d[1:, :, 3:] = velocities	
      
		for i in range(d.shape[0]):
			d[i] = normalize(d[i])
		if np.max(np.sum((np.linalg.norm(np.diff(d, axis=0), axis=2)), axis = 1)) < 25:
			test_data.append(d)

	print("Train data len: ", len(train_data))
	print("Val data len  : ", len(val_data))
	print("Test data len : ", len(val_data))
	print("Train data size: ", train_data[0].shape)
	print("Val data size  : ", val_data[0].shape)
	print("Test data size : ", test_data[0].shape)
	return train_data, val_data, test_data

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

class ActionDataset(Dataset):
	def __init__(self, data):
		self.data = data
		self.min_dim = np.min([s.shape[0] for s in self.data])
		self.max_dim = np.max([s.shape[0] for s in self.data])
		print("Min/max time steps: ", self.min_dim, self.max_dim)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

class CmuMotionData(Dataset):
    def __init__(self,
                 mode, 
                 name='cmu', 
                 data_path='', 
                 test_full=False, 
                 mask_ind_file=None):
        self.name = name
        self.data_path = data_path
        self.mode = mode
        self.train_data_len = -1
        # Get preprocessing stats.
        loc_max, loc_min, vel_max, vel_min = self._get_normalize_stats()
        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min
        self.test_full = test_full

        # Load data.
        self._load_data()
        self.expand_train = False
        if self.mode == 'train' and self.expand_train and self.train_data_len > 0:
            self.all_inds = []
            for ind in range(len(self.feat)):
                t_ind = 0
                while t_ind < len(self.feat[ind]):
                    self.all_inds.append((ind, t_ind))
                    t_ind += self.train_data_len
        else:
            self.expand_train = False

    def __getitem__(self, index):
        if self.expand_train:
            ind, t_ind = self.all_inds[index]
            start_ind = np.random.randint(t_ind, t_ind + self.train_data_len)

            feat = self.feat[ind][start_ind:start_ind + self.train_data_len]
            if len(feat) < self.train_data_len:
                feat = self.feat[ind][-self.train_data_len:]
            return {'inputs':feat}
        else: 
            inputs = self.feat[index]
            size = len(inputs)
            if self.mode == 'train' and self.train_data_len > 0 and size > self.train_data_len:
                start_ind = np.random.randint(0, size-self.train_data_len)
                inputs = inputs[start_ind:start_ind+self.train_data_len]
            result = {'inputs': inputs}
        return result

    def __len__(self, ):
        if self.expand_train:
            return len(self.all_inds)
        else:
            return len(self.feat)

    def _get_normalize_stats(self,):
        train_loc = np.load(self._get_npy_path('loc', 'train'), allow_pickle=True)
        train_vel = np.load(self._get_npy_path('vel', 'train'), allow_pickle=True)
        try:
            train_loc.max()
            self.dynamic_len = False
        except:
            self.dynamic_len = True
        if self.dynamic_len:
            max_loc = max(x.max() for x in train_loc)
            min_loc = min(x.min() for x in train_loc)
            max_vel = max(x.max() for x in train_vel)
            min_vel = min(x.min() for x in train_vel)
            return max_loc, min_loc, max_vel, min_vel
        else:
            return train_loc.max(), train_loc.min(), train_vel.max(), train_vel.min()

    def _load_data(self, ):
        #print('***Experiment hack: evaling on training.***')
        # Load data
        self.loc_feat = np.load(self._get_npy_path('loc', self.mode), allow_pickle=True)
        self.vel_feat = np.load(self._get_npy_path('vel', self.mode), allow_pickle=True)
        #self.edge_feat = np.load(self._get_npy_path('edges', self.mode))

        # Perform preprocessing.
        if self.dynamic_len:
            self.loc_feat = [normalize(feat, self.loc_max, self.loc_min) for feat in self.loc_feat]
            self.vel_feat = [normalize(feat, self.vel_max, self.vel_min) for feat in self.vel_feat]
            self.feat = [np.concatenate([loc_feat, vel_feat], axis=-1) for loc_feat, vel_feat in zip(self.loc_feat, self.vel_feat)]
            self.feat = [torch.from_numpy(np.array(feat, dtype=np.float32)) for feat in self.feat]
            print("FEATURE LEN: ",len(self.feat))
        else:
            self.loc_feat = normalize(self.loc_feat, self.loc_max, self.loc_min)
            self.vel_feat = normalize(self.vel_feat, self.vel_max, self.vel_min)

            # Reshape [num_sims, num_timesteps, num_agents, num_dims]
            #self.loc_feat = np.transpose(self.loc_feat, [0, 1, 3, 2])
            #self.vel_feat = np.transpose(self.vel_feat, [0, 1, 3, 2])
            self.feat = np.concatenate([self.loc_feat, self.vel_feat], axis=-1)

            # Convert to pytorch cuda tensor.
            self.feat = torch.from_numpy(
                np.array(self.feat, dtype=np.float32))  # .cuda()

            # Only extract the first 49 frame if testing.
            if self.mode == 'test' and not self.test_full:
                self.feat = self.feat[:, :49]

    def _get_npy_path(self, feat, mode):
        return '%s/%s_%s_%s.npy' % (self.data_path,
                                    feat,
                                    mode,
                                    self.name)

def plot_landmarks(points):
	# Separare le coordinate x e y
	x_coords = [point[0] for point in points]
	y_coords = [point[1] for point in points]
	fig = go.Figure(data=go.Scatter(x=x_coords, y=y_coords, mode='markers'))
	fig.update_layout(
		title='Grafico di coppie di punti',
		xaxis_title='Asse X',
		yaxis_title='Asse Y',
		xaxis=dict(scaleanchor='y', scaleratio=1),
		yaxis=dict(scaleanchor='x', scaleratio=1)
	)
	fig.show()

# move to base dir
print("Current dir:", os.getcwd())

if True:
    params = {}
    params['fixed_window_len'] = 40
    params['working_dir'] = 'out'
    params['num_vars'] = 42  					# num landmarks
    params['input_noise_type'] = 'none'
    params['input_size'] = 3					# num feats: pos. & vel. for each landmark
    params['input_time_steps'] = 100
    params['nll_loss_type'] = 'gaussian'
    params['prior_variance'] = 5e-2
    name = 'face_landmarks'
    params['encoder_no_factor'] = False
    params['gumbel_temp'] = 0.5
    params['num_edge_types'] = 2
    params['encoder_dropout'] = 0.05
    params['decoder_dropout'] = 0.05
    params['decoder_hidden'] = 1024
    params['encoder_hidden'] = 128
    params['encoder_rnn_hidden'] = 64
    params['encoder_mlp_hidden'] = 128
    params['encoder_rnn_type'] = 'lstm'
    params['decoder_rnn_type'] = 'gru'
    params['encoder_mlp_num_layers'] = 3
    params['prior_num_layers'] = 3
    params['prior_hidden_size'] = 128
    params['gpu'] = True
    params['continue_training'] = False
    params['skip_first'] = False
    params['load_best_model'] = False
    params['num_epochs'] = 100
    params['load_model'] = False
    params['lr'] = 5e-4
    params['lr_decay_factor'] = 0.5
    params['lr_decay_patience'] = 10
    params['lr_decay_steps'] = 300 
    params['training_scheduler'] = None
    params['step_size'] = 10
    params['batch_size'] = 4
    params['val_batch_size'] = 1
    params['accumulate_steps'] = 1
    params['verbose'] = True
    params['use_adam'] = True
    params['clean_log_dir'] = True
    params['clip_grad'] = None
    params['clip_grad_norm'] = None

# load data
path = "/home/studenti/agnelli/projects/nri/data"
train_data, val_data, test_data = load_files(path, equalize=True, eq_axis=0, target_size=100)

# make datasets
cut_size = 100
cut_size1 = 20
# train_dataset = ActionDataset(train_data[0:cut_size])
# val_dataset = ActionDataset(val_data[0:cut_size1])
# test_dataset = ActionDataset(test_data[0:cut_size1])

train_dataset = ActionDataset(train_data)
val_dataset = ActionDataset(val_data)
test_dataset = ActionDataset(test_data)

from torch.utils.data import DataLoader

params['batch_size'] = 20

# # Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False)

# # def DNRI model
# for model in ["dnri"]:
#       for graph_type in ["dynamic"]:
#         params['model_type'] = model
#         params['graph_type'] = graph_type
#         model_DNRI = model_builder.build_model(params)

#         # train DNRI model
#         params['num_epochs'] = 20
#         DO_TRAIN = True
#         if DO_TRAIN:
#             train.train(model_DNRI, train_data, val_data, params, None, None)

params['model_type'] = "dnri"
params['graph_type'] = "dynamic"
model_DNRI = model_builder.build_model(params)

# train DNRI model
params['num_epochs'] = 20
DO_TRAIN = True
if DO_TRAIN:
    train.train(model_DNRI, train_data, val_data, params, None, None)