# Graph Neural Network Daily GNSS Denoising
Repository for the daily GNSS time series denoising using a graph neural network

![overview of denoising](/assets/overview.png "Overview")

## Preprocessing
In the preprocessing folder, you will find all the scripts and notebook to prepare the data:
* 2 download script for UNR and CWU
* 2 raw to netcdf to convert and concatenate all stations together for both processing centers
* station maintenance correction notebook to remove trend using the station maintenance logs, and remove outliers
* add tremor notebook to co-locate tremors in time and space with the stations (possibility to tune the parameters)
* resource folder with the necessary files to download and add info to the stations  

Scalers and trends are saved and important to keep for rescaling the results

## Graph construction
* create_graph: main script to launch the graph construction
* GNSS_daily_clas: pytorch geometric dataset class for the GNSS dataset construction
* graphs utils: utility functions for the graph construction

## training and prediction
![GNN architecture](/assets/GNN.png "GNN architecture")
* NN_architecture: pytorch geometric GNN class
* GNN training script: automatically trains on the first n graphs of the dataset and saves the last 400 for testing, equivalent to saving 2022 and 2023 to testing in our use case.
  * Command Line Arguments:
    * 'lr' (float): Learning rate for the optimization algorithm.
    * 'hidden_layer' (int): Size of the hidden layer in the neural network.
    * 'nb_epoch' (int): Maximum number of epochs for training.
    * 'dataset_id' (str): Identifier for the dataset being used.
    * 'lambda_center' (float): Lambda parameter for the center part of the loss function.
    * 'post' (str): Additional string to add at the end of the output filename.
    * Example:
              `
              python GNN_training.py 0.001 64 100 my_dataset _experiment_1
              `
              This example sets lr=0.001, hidden_layer=64, nb_epoch=100, dataset_id='my_dataset',
              lambda_center=0.1, and post='_experiment_1'.
  * Then predicts for all the graphs in the dataset
## Results
* results_to_nc_daily: Notebook to read the graph results from the GNN and generate a netcdf
  * read graphs
  * average the overlapping windows
  * rescale and add trend
  * some simple verification figures
  * final cleanup and save
## References
TODO Add paper ref
TODO Add result dataset zenodo