import torch

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))
