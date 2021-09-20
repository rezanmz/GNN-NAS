import torch.nn.functional as F


def train(x, edge_index, labels, model, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.nll_loss(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def evaluate(x, edge_index, labels, model, validation_mask):
    model.eval()
    pred = model(x, edge_index).argmax(dim=1)
    correct = (pred[validation_mask] == labels[validation_mask]).sum()
    accuracy = int(correct) / int(validation_mask.sum())
    return accuracy
