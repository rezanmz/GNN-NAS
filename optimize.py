import matplotlib.pyplot as plt
from train import evaluate, train
from model import GraphSAGEModel
from utils import coarsen_graph, load_dataset
import optuna
import torch
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# COARSENING_RATIO = []
COARSENING_RATIO = [0.9, 0.6, 0.3]
# COARSENING_RATIO = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# COARSENING_RATIO = [0.5]
DATASET = 'CiteSeer'
PERCENT_AVAILABLE_TRAINING_DATA = 5
# Choices: ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']
COARSENING_METHOD = 'kron'


# Matplotlib stuff
plt.ion()


class DynamicUpdate():
    # Suppose we know the x range
    min_x = 0
    max_x = (len(COARSENING_RATIO) + 1) * 100

    def on_launch(self):
        # Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o', markersize=2)
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_title(
            f'Fast Optuna Optimization on {DATASET}')
        # self.ax.set_title(
        #     f'Fast Optuna Optimization using Hierarchical View {DATASET}')
        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        # self.ax.set_xlim(self.min_x, self.max_x)
        # Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Example
    # def __call__(self):
    #     import numpy as np
    #     import time
    #     self.on_launch()
    #     xdata = []
    #     ydata = []
    #     for x in np.arange(0, 10, 0.5):
    #         xdata.append(x)
    #         ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
    #         self.on_running(xdata, ydata)
    #         time.sleep(1)
    #     return xdata, ydata


d = DynamicUpdate()


# Pre-Processing
data, num_classes = load_dataset(DATASET, PERCENT_AVAILABLE_TRAINING_DATA)
x, labels, edge_index, train_mask, validation_mask = data.x, data.y, data.edge_index, data.train_mask, data.val_mask
x = x.to(device)
labels = labels.to(device)
edge_index = edge_index.to(device)
train_mask = train_mask.to(device)
validation_mask = validation_mask.to(device)

coarsened_graphs = []
for ratio in tqdm(COARSENING_RATIO, total=len(COARSENING_RATIO), desc='Generating Coarsened Graphs'):
    result = coarsen_graph(data, ratio, COARSENING_METHOD)
    # for i in range(len(result)):
    #     result[i] = result[i].to(device)
    coarsened_graphs.append({
        'ratio': ratio,
        'coarsen_x': result[0].to(device),
        'coarsen_train_labels': result[1].to(device),
        'coarsen_train_mask': result[2].to(device),
        # 'coarsen_val_labels': result[3],
        # 'coarsen_val_mask': result[4],
        'coarsen_edge': result[5].to(device),
    })

coarsened_graphs.append({
    'ratio': 0,
    'coarsen_x': x,
    'coarsen_train_labels': labels,
    'coarsen_train_mask': train_mask,
    # 'coarsen_val_labels': labels,
    # 'coarsen_val_mask': validation_mask,
    'coarsen_edge': edge_index,
})

coarsen_x = None
coarsen_train_labels = None
coarsen_train_mask = None
coarsen_edge = None

accuracies = []


def objective(trial):
    # n_layers = trial.suggest_int('n_layers', 1, 5)
    n_layers = 5
    layers = []
    for l in range(n_layers):
        layers.append({
            'output_dim': trial.suggest_int(f'l{l}_output_dim', 1, 200) if l != (n_layers - 1) else num_classes,
            'normalize': trial.suggest_categorical(f'l{l}_normalize', [True, False]),
            'root_weight': trial.suggest_categorical(f'l{l}_root_weight', [True, False]),
            'bias': trial.suggest_categorical(f'l{l}_bias', [True, False]),
            'aggr': trial.suggest_categorical(f'l{l}_aggr', ['add', 'mean', 'max']),
            'activation': trial.suggest_categorical(f'l{l}_activation', ['sigmoid', 'elu', 'relu', 'softmax', 'tanh', 'softplus', 'leaky_relu', 'relu6', None]),
            'dropout': trial.suggest_float(f'l{l}_dropout', 0.0, 1.0),
        })

    model = GraphSAGEModel(layers, x.shape[1]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters())
    for _ in range(50):
        train(coarsen_x, coarsen_edge, coarsen_train_labels,
              model, optimizer, coarsen_train_mask)
    accuracies.append(evaluate(x, edge_index, labels, model, validation_mask))
    d.on_running(range(len(accuracies)), accuracies)
    return accuracies[-1]


study = optuna.create_study(direction='maximize')

COARSENING_RATIO.append(0)

d.on_launch()

for c in COARSENING_RATIO:
    graph = None
    for coarsened_graph in coarsened_graphs:
        if coarsened_graph['ratio'] == c:
            graph = coarsened_graph
            break
    coarsen_x = graph['coarsen_x']
    coarsen_train_labels = graph['coarsen_train_labels']
    coarsen_train_mask = graph['coarsen_train_mask']
    coarsen_edge = graph['coarsen_edge']
    print('Graph Size:', coarsen_x.shape[0])
    study.optimize(objective, n_trials=50, show_progress_bar=True)

input()
