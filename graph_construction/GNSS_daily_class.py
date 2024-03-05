import numpy as np
import torch
import xarray as xr
from torch_geometric.data import InMemoryDataset, Data
from time import time
import graphs_utils


class GNSSDaily(InMemoryDataset):
    """
    Custom PyTorch Geometric dataset for GNSS (Global Navigation Satellite System) data.

    Args:
        root: Root directory for storing the processed dataset.
        transform: Data transformation to be applied during processing.
        pre_transform: Pre-processing transformation (e.g., KNNGraph).
        path_ds: Path to the input netCDF dataset.
        edge_length: Maximum edge distance for pruning graph edges.
        nb_edges: Number of edges to retain for each node after pruning.
    """

    def __init__(self, root, transform=None, pre_transform=None, path_ds="./train_clean.nc", edge_length=400,
                 nb_edges=10):
        """
        Initialize the GNSSDaily dataset.

        Args are the same as the class constructor.
        """
        self.path_ds = path_ds
        self.window_size = 30
        self.nb_edges = nb_edges
        self.edge_length = edge_length
        super(GNSSDaily, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        List of raw files used by the dataset.
        """
        return 'subset.nc'

    @property
    def processed_file_names(self):
        """
        List of processed files for the dataset.
        """
        return 'data.pt'

    def process(self):
        """
        Process the raw dataset and store the processed data.
        """
        # Read the file netcdf and create dataset
        ds = xr.open_dataset(self.path_ds).load()

        data_list = []
        offset = self.window_size // 10

        for i in range(len(ds.time.data) // offset):
            ds_tmp = ds.isel(time=slice(i * offset, (i * offset) + self.window_size))
            ds_tmp = ds_tmp.dropna(dim="station", thresh=int(self.window_size * 0.8), subset=['e_norm'])
            if len(ds_tmp.station) > 10 and len(ds_tmp.time) == self.window_size:
                ds_tmp = ds_tmp.fillna(0)
                g = Data(pos=torch.tensor(np.array([ds_tmp.longitude.data[()], ds_tmp.latitude.data[()]]).T).type(
                    torch.FloatTensor))
                g.id = ds_tmp.station.data[()]
                g.date_start = ds_tmp.time[0].data[()]
                g.cos_day = torch.full((len(ds_tmp.station),), np.cos(ds_tmp.time.dt.dayofyear[0]).item()).type(
                    torch.FloatTensor)
                g.sin_day = torch.full((len(ds_tmp.station),), np.sin(ds_tmp.time.dt.dayofyear[0]).item()).type(
                    torch.FloatTensor)
                # g.elevation = torch.tensor(ds_tmp.height.data[()]).type(torch.FloatTensor)
                g.signal_n = torch.tensor(ds_tmp.n_norm.data.T).type(torch.FloatTensor)
                g.signal_e = torch.tensor(ds_tmp.e_norm.data.T).type(torch.FloatTensor)
                g.signal_z = torch.tensor(ds_tmp.z_norm.data.T).type(torch.FloatTensor)

                g.node_degree = torch.tensor(np.zeros(len(ds_tmp.station))).type(torch.FloatTensor)
                data_list.append(g)

        print('done creating all the graphs')
        if self.pre_transform is not None:
            start_time = time()
            data_list = [self.pre_transform(data) for data in data_list]
            end_time = time()
            print(f"computed knn in {end_time - start_time}")
            start_time = time()
            data_list = [graphs_utils.add_edge_dist(data) for data in data_list]
            end_time = time()
            print(f"computed dists in {end_time - start_time}")
            start_time = time()
            data_list = [graphs_utils.prune_graph(data, self.nb_edges, self.edge_length) for data in data_list]
            end_time = time()
            print(f"computed pruning in {end_time - start_time}")

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
