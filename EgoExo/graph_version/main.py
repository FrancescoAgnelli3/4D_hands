from tqdm import tqdm, trange
from dataset import Ego4D
import torch_geometric.transforms as T
from utility import transforms as U
from utility.train_eval import train, eval
from torch_geometric.loader import DataLoader
import torch
from model.Euler_method import TemporalGraphEuler
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(0)

path = "/mnt/wwn-0x5000000000000001-part1/datasets/EgoExo4d/data_train"
dataset_train = Ego4D(
    root=path,
    )

# dataset_train.apply_transform_to_dataset(U.task())

path_val = "/mnt/wwn-0x5000000000000001-part1/datasets/EgoExo4d/data_val"
dataset_val = Ego4D(
    root=path_val,
    )

# dataset_val.apply_transform_to_dataset(U.task())


# print("Padding...")
# length = 0
# for data in dataset_val:
#     length = np.max([length, len(data["right_land"])])
# for data in dataset_train:
#     length = np.max([length, len(data["right_land"])])

# dataset_val.apply_transform_to_dataset(U.padding(length))
# dataset_train.apply_transform_to_dataset(U.padding(length))

# calculate spline coefficients

# dataset_train.apply_transform_to_dataset(U.SplineCoeff())
# dataset_val.apply_transform_to_dataset(U.SplineCoeff())

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TemporalGraphEuler(
    input_dim=3,
    output_dim=12,
    delta_t=5,
    hidden_dim=128
)
model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# train
model.train()
for epoch in trange(101):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scheduler)
    test_loss, test_acc = eval(model, test_loader, criterion, device)

    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")