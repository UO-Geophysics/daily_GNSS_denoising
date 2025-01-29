from torch_geometric.transforms import KNNGraph
from GNSS_daily_class import GNSSDaily


def main():
    """
    Main function for executing the GNSSDaily dataset processing.
    """
    edge_length = 400
    nb_edges = 8
    path_to_ds = "../data/clean_daily_UNR.nc"
    dataset = GNSSDaily(root=f"../data/graph_UNR_{edge_length}km_k{nb_edges}_30day_final",
                        pre_transform=KNNGraph(k=300, loop=False, force_undirected=True),
                        path_ds=path_to_ds, edge_length=edge_length, nb_edges=nb_edges)
    print(dataset[0])


if __name__ == '__main__':
    main()
