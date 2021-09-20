import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import activation
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, layers, feat_size) -> None:
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in layers:
            if len(self.convs) == 0:
                input_dim = feat_size
            self.convs.append(SAGEConv(
                in_channels=input_dim,
                out_channels=layer['output_dim'],
                normalize=layer['normalize'],
                root_weight=layer['root_weight'],
                bias=layer['bias'],
                aggr=layer['aggr'],
            ))
            if layer['activation'] == 'sigmoid':
                self.convs.append(activation.Sigmoid())
            elif layer['activation'] == 'elu':
                self.convs.append(activation.ELU())
            elif layer['activation'] == 'relu':
                self.convs.append(activation.ReLU())
            elif layer['activation'] == 'softmax':
                self.convs.append(activation.Softmax())
            elif layer['activation'] == 'tanh':
                self.convs.append(activation.Tanh())
            elif layer['activation'] == 'softplus':
                self.convs.append(activation.Softplus())
            elif layer['activation'] == 'leaky_relu':
                self.convs.append(activation.LeakyReLU())
            elif layer['activation'] == 'relu6':
                self.convs.append(activation.ReLU6())

            self.convs.append(nn.Dropout(p=layer['dropout']))

            input_dim = layer['output_dim']

    def forward(self, x, edge_index):
        for layer in self.convs:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        return F.log_softmax(x, dim=1)
