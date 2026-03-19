"""Train and apply a simple feedforward network for predicting LAI.

This module defines:
- MyDataset: a PyTorch Dataset that reads multiple CSV files containing
  columns ['time', 't2m', 'tp', 'ssrd', 'swvl1', 'swvl2', 'swvl3', 'swvl4',
  'lat', 'lon', 'lai']. It concatenates all files, stores feature and
  target tensors, and computes per-feature and target normalization
  statistics.
- SimpleNet: a small fully connected neural network with two hidden
  layers (64 and 32 units) and ReLU activations, mapping input feature
  vectors to a scalar LAI prediction.
- A __main__ block that constructs datasets and dataloaders, trains the
  model with MSE loss and SGD, evaluates on a test split, and performs an
  "upscaling" step that writes predictions into a 3D grid and plots the
  results.

__author__ = Christian Reimers (creimers@bgc-jena.mpg.de)
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn, optim

print(torch.cuda.is_available())

class MyDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads CSV files and provides normalized samples.

    Each CSV file must contain the columns:
    ['time', 't2m', 'tp', 'ssrd', 'swvl1', 'swvl2', 'swvl3', 'swvl4',
    'lat', 'lon', 'lai'].

    The class concatenates all files into single tensors for inputs and
    targets and computes per-feature and per-target statistics.

    Attributes:
        data (torch.Tensor): Tensor of shape (N, 10) with input features.
        targets (torch.Tensor): Tensor of shape (N,) with target values.
        data_mean (torch.Tensor): Mean of data per feature, shape (1, 10).
        data_std (torch.Tensor): Std of data per feature, shape (1, 10).
        targets_mean (torch.Tensor): Mean of targets, shape (1, 1).
        targets_std (torch.Tensor): Std of targets, shape (1, 1).

    Notes:
        __getitem__ normalizes both inputs and targets using the target
        mean/std (this matches the original script behaviour).

    """

    def __init__(self, files: list[str]) -> None:
        """Initialize dataset by reading and concatenating CSV files.

        Args:
            files: List of file paths (strings) to CSV files.

        """
        self.data: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        for file in files:
            data = pd.read_csv(file)

            time = np.array(data["time"])
            t2m = np.array(data["t2m"])
            tp = np.array(data["tp"])
            ssrd = np.array(data["ssrd"])
            swvl1 = np.array(data["swvl1"])
            swvl2 = np.array(data["swvl2"])
            swvl3 = np.array(data["swvl3"])
            swvl4 = np.array(data["swvl4"])
            lat = np.array(data["lat"])
            lon = np.array(data["lon"])

            x = np.stack(
                [t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, time, lat, lon],
                axis=-1,
            )
            y = np.array(data["lai"])

            self.data.append(torch.Tensor(x))
            self.targets.append(torch.Tensor(y))
    
        ########################################################################
        # concatenate into single tensors
        self.data = torch.concatenate(self.data, dim=0)
        self.targets = torch.concatenate(self.targets, dim=0)
        self.data = self.data.reshape(-1, 24, 10)
        self.targets = self.targets.reshape(-1, 24, 1)
        ########################################################################

        # compute normalization statistics
        self.data_mean: torch.Tensor = self.data.mean(dim=0, keepdims=False)
        self.data_std: torch.Tensor = self.data.std(dim=0, keepdims=False)
        self.targets_mean: torch.Tensor = self.targets.mean(
            dim=0, keepdims=False,
        )
        self.targets_std: torch.Tensor = self.targets.std(dim=0, keepdims=False)

    def __len__(self) -> int:
        """Return number of samples."""
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """Return normalized (input, target) pair for a given index.

        Both input and target are normalized using the dataset's
        targets_mean and targets_std (mirrors original script).

        Args:
            index: Sample index.

        Returns:
            Tuple (input_tensor, target_tensor) both of type torch.Tensor.

        """
        return (self.data[index] - self.targets_mean) / self.targets_std, (
            self.targets[index] - self.targets_mean
        ) / self.targets_std


class SimpleNet(nn.Module):
    """Simple feedforward neural network.

    This module is intended for small regression tasks that map a
    feature vector to a single (or few) continuous outputs.

    Args:
        in_dim: Number of input features.
        out_dim: Number of output features.

    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # This network now expects input of the shape (batchsize, 24, 10) where
        # second dimension is the time in the year. Feel free to play around 
        # with it
        #=======================================================================
        self.lin1: nn.Linear = nn.Linear(24 * in_dim, 64)
        self.lin2: nn.Linear = nn.Linear(64, 64)
        self.lin3: nn.Linear = nn.Linear(64, 24 * out_dim)

        self.relu: nn.ReLU = nn.Tanh()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        #=======================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape (..., in_dim).

        Returns:
            Output tensor of shape (..., out_dim).

        """
        # This network now expects input of the shape (batchsize, 24, 10) where
        # second dimension is the time in the year. Feel free to play around 
        # with it
        #=======================================================================
        x = x.view(-1, 24 * self.in_dim)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        return self.lin3(x).reshape(-1, 24, self.out_dim)
        #=======================================================================

if __name__ == "__main__":
    device = "cpu"
    
    #consider adapting the number of epochs
    epochs = 10
    learning_rate = 3e-5
    input_dim = 10
    output_dim = 1
    model = SimpleNet(input_dim, output_dim)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_dataset = MyDataset([f"../data/sites_{i}.csv" for i in range(80)])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True,
    )

    test_dataset = MyDataset([f"../data/sites_{i}.csv" for i in range(80, 100)])
    test_dataset.data_mean = train_dataset.data_mean
    test_dataset.data_std = train_dataset.data_std
    test_dataset.targets_mean = train_dataset.targets_mean
    test_dataset.targets_std = train_dataset.targets_std
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True,
    )

    # Training loop
    progress_bar = tqdm.tqdm(range(1, epochs + 1))
    for _ in progress_bar:
        model.train()
        running_loss = 0.0
        for xb, yb in train_dataloader:

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_dataset)
        progress_bar.set_postfix({"loss": epoch_loss})

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            predictions.append(pred.detach().numpy())
            labels.append(y.detach().numpy())

    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)

    plt.figure()
    plt.scatter(labels, predictions)
    plt.show()
    plt.close()

    print(1 - np.var(predictions - labels) / np.var(labels))
    print(np.mean((labels - predictions)**2))

    data = pd.read_csv("../data/upscaling.csv")

    time = np.array(data["time"])
    t2m = np.array(data["t2m"])
    tp = np.array(data["tp"])
    ssrd = np.array(data["ssrd"])
    swvl1 = np.array(data["swvl1"])
    swvl2 = np.array(data["swvl2"])
    swvl3 = np.array(data["swvl3"])
    swvl4 = np.array(data["swvl4"])
    lat = np.array(data["lat"])
    lon = np.array(data["lon"])
    coord_x = np.array(data["x"])
    coord_y = np.array(data["y"])

    coords = np.stack([time // 15, coord_x, coord_y], axis=-1)
    upscale_x = torch.Tensor(
        np.stack(
            [t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, time, lat, lon], axis=-1,
        ),
    ).reshape(-1,24,10)
    upscale_y = np.array(data["lai"]).reshape(-1,24)

    with torch.no_grad():
        preds = model(upscale_x).squeeze().detach().cpu().numpy().astype(float)
        preds = preds * train_dataset.targets_std.cpu().numpy().T + train_dataset.targets_mean.cpu().numpy().T

    print(preds.shape)
    print(upscale_y.shape)
    print('R2:', 1 - np.var(preds - upscale_y) / np.var(upscale_y))
    print('MSE:', np.mean((upscale_y - preds)**2))
   

    preds = preds.reshape(-1)
    upscale_y = upscale_y.reshape(-1)
    output = np.ones((24, 146, 837 - 678)) * float("NaN")
    output_true = np.ones((24, 146, 837 - 678)) * float("NaN")
    for nr in range(coords.shape[0]):
        output[*coords[nr]] = preds[nr]
        output_true[*coords[nr]] = upscale_y[nr]


    

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(output.mean(0), cmap="Greens")
    plt.subplot(1, 3, 2)
    plt.imshow(output_true.mean(0), cmap="Greens")
    plt.subplot(1, 3, 3)
    plt.imshow(output_true.mean(0) - output.mean(0), cmap="coolwarm")
    plt.show()
    plt.close()
