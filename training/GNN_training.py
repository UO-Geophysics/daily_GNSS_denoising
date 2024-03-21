import logging
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import KNNGraph
from torch.nn import MSELoss
from NN_architecture import MLPNet
from ..graph_construction.GNSS_daily_class import GNSSDaily
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: An object containing parsed command line arguments.

    Command Line Arguments:
        - 'lr' (float): Learning rate for the optimization algorithm.
        - 'hidden_layer' (int): Size of the hidden layer in the neural network.
        - 'nb_epoch' (int): Maximum number of epochs for training.
        - 'dataset_id' (str): Identifier for the dataset being used.
        - 'post' (str): Additional string to add at the end of the output filename.

    Example:
        ```bash
        python script.py 0.001 64 100 my_dataset _experiment_1
        ```
        This example sets lr=0.001, hidden_layer=64, nb_epoch=100, dataset_id='my_dataset',
        and post='_experiment_1'.
    """
    import argparse

    parse = argparse.ArgumentParser(description="")
    parse.add_argument('lr', type=float, help="learning rate")
    parse.add_argument('hidden_layer', type=int, help="size hidden layer")
    parse.add_argument('nb_epoch', type=int, help="max number epoch")
    parse.add_argument('dataset_id', type=str, help="dataset id")
    parse.add_argument('post', type=str, help="post str to add at the end of filename")
    return parse.parse_args()


class EarlyStopper:
    """
    A class for early stopping the training process when the validation loss stops improving.

    Parameters:
    -----------
    patience : int, optional (default=1)
        The number of epochs with no improvement in validation loss after which training will be stopped.
    min_delta : float, optional (default=0)
        The minimum change in the validation loss required to qualify as an improvement.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """
        Check if the training process should be stopped.

        Parameters:
        -----------
        validation_loss : float
            The current validation loss.

        Returns:
        --------
        stop : bool
            Whether the training process should be stopped or not.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, dataloader, device, optimizer, criterion, label_rate=0.3):
    """
    Train the neural network model using labeled data.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing training data.
        device (torch.device): Device on which to perform training (e.g., 'cuda' or 'cpu').
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        criterion (torch.nn.Module): Loss function for training.
        label_rate (float): Rate of labeled data used for training.

    Returns:
        float: Mean training loss.

    Example:
        ```python
        train_loss = train(my_model, train_dataloader, device, my_optimizer, my_criterion, label_rate=0.5)
        ```

    Note:
        The `model` should be in training mode before calling this function.
    """
    model.train()
    mean_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        signal_in = torch.stack([batch.signal_n, batch.signal_e, batch.signal_z])
        signal_in = torch.flatten(torch.permute(signal_in, (1, 0, 2)), start_dim=1)
        mask = MLPNet.get_mask(signal_in.shape[0], label_rate, device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(signal_in, batch.edge_index, batch.edge_dist, mask)
        loss = criterion(out[mask], signal_in[mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        mean_loss += loss
    return float(mean_loss / len(dataloader)) * label_rate


@torch.no_grad()
def validation(model, dataloader, device, criterion, label_rate=0.3):
    """
    Evaluate the performance of the neural network model on validation data.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing validation data.
        device (torch.device): Device on which to perform validation (e.g., 'cuda' or 'cpu').
        criterion (torch.nn.Module): Loss function for validation.
        label_rate (float): Rate of labeled data used for validation.

    Returns:
        float: Mean validation loss.

    Example:
        ```python
        val_loss = validation(my_model, val_dataloader, device, my_criterion, label_rate=0.5)
        ```

    Note:
        The `model` should be in evaluation mode before calling this function.
    """
    model.eval()
    loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        signal_in = torch.stack([batch.signal_n, batch.signal_e, batch.signal_z])
        signal_in = torch.flatten(torch.permute(signal_in, (1, 0, 2)), start_dim=1)
        mask = MLPNet.get_mask(signal_in.shape[0], label_rate, device)

        out = model(signal_in, batch.edge_index, batch.edge_dist, mask)
        loss += criterion(out[mask], signal_in[mask])
    return float(loss / len(dataloader)) * label_rate


def main():
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("training_MLP.log"),
            logging.StreamHandler()
        ]
    )
    # Parse command line arguments
    args = get_args()
    lr = args.lr
    hidden_layer = args.hidden_layer
    nb_epoch = range(args.nb_epoch)
    dataset_id = args.dataset_id
    lambda_center = args.lambda_center
    post = args.post

    logging.info(
        f"learning rate = {lr}, hidden layer size = {hidden_layer}, dataset = {dataset_id}, lambda={lambda_center}, post={post}")
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"runing on {device}, {torch.cuda.device_count()} gpu , name {torch.cuda.current_device()}")

    # Load training dataset
    dataset = GNSSDaily(root=f"../graph_{dataset_id}",
                        pre_transform=KNNGraph(k=5, loop=False, force_undirected=True),
                        path_ds=f"../data/clean_daily_train.nc")

    # Split dataset into training and validation sets
    logging.info(f"First graph in testing dataset = {dataset[-400]['date_start']}")
    train_dataset, val_dataset = train_test_split(dataset[:-400], train_size=0.8, random_state=42)
    logging.info(f"Number graph in training dataset = {len(train_dataset)}")
    logging.info(f"Number graph in validation dataset = {len(val_dataset)}")

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    data = dataset[0]
    signal_shape = data['signal_n'].shape[1] * 3
    # Initialize MLPNet model
    model = MLPNet(signal_shape, hidden_channels=hidden_layer)

    # training settings
    early_stopper = EarlyStopper(patience=50, min_delta=0.0)
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    loss_train = []
    loss_val = []
    best_loss = 100
    best_epoch = 0
    PATH_CHECKPOINT = f"../checkpoint/mlp_{dataset_id}_{post}.pt"
    start_time = time.perf_counter()

    logging.info(f"training starting, nb epoch={len(nb_epoch)}")
    for epoch in nb_epoch:
        loss_train.append(train(model, train_loader, device, optimizer, criterion))
        loss_val.append(validation(model, val_loader, device, criterion))

        # Save model checkpoint if validation loss improves
        if loss_val[-1] < best_loss:
            best_loss = loss_val[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), PATH_CHECKPOINT)

        # Early stopping check
        if early_stopper.early_stop(loss_val[-1]):
            print(f"early stopping at epoch {epoch}: train loss={loss_train[-1]}, val loss={loss_val[-1]}")
            break

        # Logging training progress
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: train loss={loss_train[-1]:.6f}, val loss={loss_val[-1]:.6f}")
            logging.info(f"Best loss={best_loss} at epoch {best_epoch}")
            logging.info(f"average time per epoch {(time.perf_counter() - start_time) / 10:.2f}")
            start_time = time.perf_counter()

    # Finalize training
    logging.info(f"training finished, restoring best weights")
    model.load_state_dict(torch.load(PATH_CHECKPOINT, map_location='cuda:0'))

    # Verify model results on validation dataset
    logging.info(
        f"best loss={best_loss}, model eval loss={validation(model, val_loader, device, criterion)} at epoch {best_epoch}")

    logging.info("starting evaluation on test dataset")

    logging.info(f'Getting results for full dataset')
    logging.info(f"dataset path = ../graph_{dataset_id}")
    dataset_test = GNSSDaily(root=f"../graph_{dataset_id}",
                             pre_transform=KNNGraph(k=5, loop=False, force_undirected=True),
                             path_ds="../data/clean_daily_test.nc")
    results = []
    # Evaluate model on each data point in the full dataset
    for data in dataset_test:
        data = data.to(device)
        signal_in = torch.stack([data.signal_n, data.signal_e, data.signal_z])
        signal_in = torch.flatten(torch.permute(signal_in, (1, 0, 2)), start_dim=1)
        mask = MLPNet.get_mask(signal_in.shape[0], 0, device)
        data.n_out, data.e_out, data.z_out = torch.unbind(
            model(signal_in, data.edge_index, data.edge_dist, mask).detach().reshape(-1, 3, signal_shape // 3), dim=-2)
        results.append(data)
    # Logging and saving results
    logging.info(f"evaluation finished")
    logging.info(f"Saving the results in ./results_{dataset_id}_{post}.pt")
    torch.save(results, f'results_{dataset_id}_{post}.pt')
    logging.info("training and evaluation finished")
    return 0


if __name__ == '__main__':
    main()
