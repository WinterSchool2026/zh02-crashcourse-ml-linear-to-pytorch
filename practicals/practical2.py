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

        # concatenate into single tensors
        self.data = torch.concatenate(self.data, dim=0)
        self.targets = torch.concatenate(self.targets, dim=0)

        # compute normalization statistics
        self.data_mean: torch.Tensor = self.data.mean(dim=0, keepdims=True)
        self.data_std: torch.Tensor = self.data.std(dim=0, keepdims=True)
        self.targets_mean: torch.Tensor = self.targets.mean(
            dim=0, keepdims=True,
        )
        self.targets_std: torch.Tensor = self.targets.std(dim=0, keepdims=True)

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
        #=======================================================================
        # create the layers that you need with nn.Linear(dim_in, dim_out)
       



        # decide for a non-linearity (nn.ReLU() or nn.Tanh()) or something else
        
        #=======================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape (..., in_dim).

        Returns:
            Output tensor of shape (..., out_dim).

        """
        #=======================================================================
        # Call the layers with the non-linearity in between
        



        return x
        #=======================================================================



if __name__ == "__main__":
    # use a gpu if one is available. If there are errors mentoning cuda comment
    # this line out and use 
    # device = 'cpu'
    # instead
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # the number of epochs is how often each data point is shown to the neural 
    # network
    epochs = 10
    # learning rate is the step size of the optimizer
    learning_rate = 3e-5
    # the number of input variables and output variables
    input_dim = 10
    output_dim = 1

    #===========================================================================
    #build your model by calling model = ...



    # create a loss function
    
    # create an optimizer
    
    #===========================================================================

    # create your dataset for train and test data
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
    # iterate over the epochs
    for _ in progress_bar:
        # put the model in training mode
        model.train()
        running_loss = 0.0

        # loop over all examples in the training data
        for xb, yb in train_dataloader:
            #===================================================================        
            # reset the optimizer

            # make predictions with the model

            # calculate the loss

            # calculate the gradients comming from the loss

            # make a step with the optimizer along the gradient

            #===================================================================
            # sum up the loss
            running_loss += loss.item() * xb.size(0)

        # output the loss after each epoch
        epoch_loss = running_loss / len(train_dataset)
        progress_bar.set_postfix({"loss": epoch_loss})

    # store and load the model. It does not make sense here, I just wanted you
    # to know how to do it :D
    # save
    torch.save(model.state_dict(), "model_weights.pth")
    
    #to load, you first need to initialize the model and then load the weights
    model = SimpleNet(input_dim, output_dim)
    model.load_state_dict(torch.load("model_weights.pth"))
    



    # put the model into evaluation mode
    model.eval()
    predictions = []
    labels = []
    # do not calculate gradients
    with torch.no_grad():
        # loop over all the examples in the test set
        for X, y in test_dataloader:
            #make a prediction using the model
            pred = model(X)

            #move the data to cpu and to numpy
            predictions.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    # make one array out of all the predictions and out of all the labels
    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)

    # scatterplot of predictions and labels
    plt.figure()
    plt.scatter(labels, predictions, color = '#005555')
    plt.xlabel('labels (normalized)')
    plt.ylabel('predictions (normalized)')
    plt.show()
    plt.close()

    # calculate the R2 and the MSE
    print(1 - np.var(predictions - labels) / np.var(labels))
    print(np.mean((labels - predictions)**2))


    # load data for upscaling
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

    # store coordinates and create data and labels
    coords = np.stack([time // 15, coord_x, coord_y], axis=-1)
    upscale_x = torch.Tensor(
        np.stack(
            [t2m, tp, ssrd, swvl1, swvl2, swvl3, swvl4, time, lat, lon], axis=-1,
        ),
    )
    upscale_y = np.array(data["lai"])
    
    # make sure to not calculate gradients in inference
    with torch.no_grad():
        # make the predictions and denormalize them
        preds = model(upscale_x).squeeze().detach().cpu().numpy().astype(float)
        preds = preds * train_dataset.targets_std.cpu().numpy() + train_dataset.targets_mean.cpu().numpy()

    # print the R2 and the mean squarred error
    print('R2:', 1 - np.var(preds - upscale_y) / np.var(upscale_y))
    print('MSE:', np.mean((upscale_y - preds)**2))
    
    # sort the outputs to the correct locations
    output = np.ones((24, 146, 837 - 678)) * float("NaN")
    output_true = np.ones((24, 146, 837 - 678)) * float("NaN")
    for nr in range(coords.shape[0]):
        output[*coords[nr]] = preds[nr]
        output_true[*coords[nr]] = upscale_y[nr]

    # plot the prediction and the difference to the label
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(output.mean(0), cmap="Greens")
    plt.subplot(1, 3, 2)
    plt.imshow(output_true.mean(0), cmap="Greens")
    plt.subplot(1, 3, 3)
    plt.imshow(output_true.mean(0) - output.mean(0), cmap="coolwarm")
    plt.show()
    plt.close()
