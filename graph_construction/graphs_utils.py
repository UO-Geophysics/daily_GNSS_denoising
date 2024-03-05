import numpy as np
import torch
from numba import njit


@njit
def haversine_nb(lon1, lat1, lon2, lat2):
    """
    Calculate Haversine distance between two sets of latitude and longitude coordinates.

    Args:
        lon1, lat1: Longitude and latitude of the first point.
        lon2, lat2: Longitude and latitude of the second point.

    Returns:
        Haversine distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


def add_edge_dist(g):
    """
    Add edge distances to a graph based on Haversine distance between node positions.

    Args:
        g: Input graph with node positions.

    Returns:
        g: Graph with added edge distances.
    """
    pos = g.pos.numpy()
    edges = g.edge_index.numpy()
    dist = haversine_nb(pos[edges[0]].T[1], pos[edges[0]].T[0], pos[edges[1]].T[1], pos[edges[1]].T[0])
    g.edge_dist = torch.tensor(dist).type(torch.FloatTensor)
    return g


def prune_graph(g, nb_edges, edge_length):
    """
    Prune graph edges based on distance and retain the top 'nb_edges' edges for each node.

    Args:
        g: Input graph with edge distances.
        nb_edges: Number of edges to retain for each node.
        edge_length: Maximum edge distance to retain.

    Returns:
        g: Pruned graph.
    """
    pop_index_list = (g.edge_dist > edge_length)
    g.edge_dist = g.edge_dist[pop_index_list]
    g.edge_index = g.edge_index.T[pop_index_list].T
    indices = []
    for i in range(len(g.id)):
        try:
            start = torch.argwhere(g.edge_index.T[:, 0] == i)[0][0]
            indices.append(torch.topk(g.edge_dist[g.edge_index.T[:, 0] == i],
                                      min(nb_edges, g.edge_dist[g.edge_index.T[:, 0] == i].shape[0]),
                                      largest=False).indices + start)
        except:
            print(f"problem with node {i} or graph {g.date_start}")
    indices = torch.cat(indices, 0)
    g.edge_dist = g.edge_dist[indices]
    g.edge_index = g.edge_index.T[indices].T
    return g