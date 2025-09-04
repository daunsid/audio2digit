import torch


def accuracy_sc(outputs: torch.Tensor, labels: torch.Tensor):
    indices = torch.argmax(outputs, dim=1)
    correct_preds = (indices == labels).sum().item()
    return correct_preds/labels.size(0)