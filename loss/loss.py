import torch

def elementwise_max(tensor, scalar):
  """
  Lay function on top of torch.max to get behavior like
  torch.min(tensor1, 0.0)
  :param tensor: The tensor you want the elements compared to
  :param scalar: The number to use in max.
  :return:
  """
  tensor2 = torch.randn(tensor.shape)
  tensor2.fill_(scalar)
  tensor2 = tensor2.cuda()
  return torch.max(tensor, tensor2)

def MAPELoss(output, target):
  eps = 1e-07
  diff = torch.abs((target - output) / elementwise_max(torch.abs(target), eps))
  loss = 100. * diff.mean() #get average to reduce tensor size
  return loss

"""
  diff = math_ops.abs(
      (y_true - y_pred) / K.maximum(math_ops.abs(y_true), K.epsilon()))
  return 100. * K.mean(diff, axis=-1)
"""

if __name__ == "__main__":
  x = torch.tensor([[1, 1, 1]], dtype=torch.float)
  y = torch.tensor([[1.5, 1.5, 1.5]], dtype=torch.float)
  print(MAPELoss(x, y))