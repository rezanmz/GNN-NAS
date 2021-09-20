import torch
from tqdm import tqdm

from model import GraphSAGEModel
from train import evaluate, train
from utils import coarsen_graph, load_dataset

data, num_classes = load_dataset('PubMed', 5)

x, labels, edge_index, train_mask, validation_mask = data.x, data.y, data.edge_index, data.train_mask, data.val_mask

coarsen_x, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = coarsen_graph(
    data, 0.5, 'kron')
print('Size of the original graph:', x.shape[0])
print('Size of the coarsened graph:', coarsen_x.shape[0])

# raise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

x = x.to(device)
labels = labels.to(device)
edge_index = edge_index.to(device)
train_mask = train_mask.to(device)
validation_mask = validation_mask.to(device)

coarsen_x = coarsen_x.to(device)
coarsen_train_labels = coarsen_train_labels.to(device)
coarsen_train_mask = coarsen_train_mask.to(device)
coarsen_val_labels = coarsen_val_labels.to(device)
coarsen_val_mask = coarsen_val_mask.to(device)
coarsen_edge = coarsen_edge.to(device)

train_type = 'coarsened'
# train_type = 'original'

layers = [{
    'output_dim': 128,
    'normalize': True,
    'root_weight': True,
    'bias': True,
    'aggr': 'mean',
    'activation': 'relu',
    'dropout': 0,
}, {
    'output_dim': num_classes,
    'normalize': True,
    'root_weight': True,
    'bias': True,
    'aggr': 'mean',
    'activation': 'relu',
    'dropout': 0,
}]
model = GraphSAGEModel(layers, x.shape[1]).to(device)
optimizer = torch.optim.RMSprop(model.parameters())
if train_type == 'coarsened':
    losses = []
    accuracies = []
    pbar = tqdm(range(50), total=50, desc='Accuracy: Inf')
    for i in pbar:
        train(coarsen_x, coarsen_edge, coarsen_train_labels,
              model, optimizer, coarsen_train_mask)
        pbar.set_description_str(
            'Accuracy: ' + str(evaluate(x, edge_index, labels, model, validation_mask)))
    for a in accuracies:
        print(a)
else:
    losses = []
    accuracies = []
    pbar = tqdm(range(50), total=50, desc='Accuracy: Inf')
    for i in pbar:
        train(x, edge_index, labels, model, optimizer, train_mask)
        pbar.set_description_str(
            'Accuracy: ' + str(evaluate(x, edge_index, labels, model, validation_mask)))
    for a in accuracies:
        print(a)
