import torch
from torch import nn
from torch_geometric.nn import MLP, GATv2Conv, Linear
    
    
class MLPNet(torch.nn.Module):
    """
    Implementation of a neural network model (MLPNet) for graph-based data using GATv2Conv.

    Args:
        channels_y (int): Number of output channels.
        hidden_channels (int): Number of hidden channels in the GAT convolutional layer.
        dropout (float): Dropout probability for regularization.
        self_loops (bool): Whether to include self-loops in the GAT convolutional layer.

    Attributes:
        mlp_in (MLP): MLP module for initial processing of input signals.
        conv (GATv2Conv): GATv2Conv layer for graph attention mechanism.
        mlp_out (MLP): MLP module for processing the output of the GAT layer.
        out_linear (Linear): Linear layer for the final output.

    Methods:
        forward(signal, edge_index, edge_weight, mask):
            Performs forward pass through the network.

        get_mask(graph_size, rate, device):
            Generates a binary mask for graph nodes based on dropout rate.

    Example:
        model = MLPNet(channels_y=64, hidden_channels=10, dropout=0.3, self_loops=False)
        output = model(input_signal, edge_index, edge_weight, mask)
    """
    def __init__(self, channels_y, hidden_channels=10, dropout=0.3, self_loops=False):
        super(MLPNet, self).__init__()
        torch.manual_seed(1234)
        self.mlp_in = MLP([channels_y, 512, hidden_channels])
        nb_heads = 2
        self.conv = GATv2Conv(hidden_channels, hidden_channels, edge_dim=1, heads=nb_heads, add_self_loops=self_loops)
        self.mlp_out = MLP([hidden_channels*nb_heads, 512], act=nn.LeakyReLU(), dropout=dropout)
        self.out_linear = Linear(512, channels_y)
    
    def forward(self, signal, edge_index, edge_weight, mask):
        """
        Forward pass through the MLPNet model.

        Args:
            signal (Tensor): Input signal features for each node.
            edge_index (LongTensor): Graph edge indices.
            edge_weight (Tensor): Edge weights for each edge.
            mask (Tensor): Binary mask for dropout.

        Returns:
            out (Tensor): Model output.
        """
        out = (signal.T*(~mask)).T
        # MLP in before message passing
        out = self.mlp_in(out)
        
        # 1 layers message passing
        out = self.conv(out, edge_index, edge_weight)
        
        # MLP before decoder (has dropout)
        out = self.mlp_out(out)
        out = self.out_linear(out)
        return out
    
    @staticmethod
    def get_mask(graph_size: int, rate: float, device):
        """
        Generate a binary mask for graph nodes based on dropout rate.

        Args:
            graph_size (int): Number of nodes in the graph.
            rate (float): Dropout rate.
            device (torch.device): Device on which to create the mask.

        Returns:
            mask (Tensor): Binary mask with dropout applied.
        """
        mask = torch.rand(graph_size, device=device) < rate
        return mask
