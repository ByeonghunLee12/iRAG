from torch.autograd import Function
import torch
import torch.nn as nn

#### Hash Functions ####
class HashFunction(Function):
    """
    Custom hash function with a straight-through estimator.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

class ContinuousFunction(nn.Module):
    def __init__(self):
        super(ContinuousFunction, self).__init__()
    
    def forward(self, x):
        return torch.sin(torch.atan(x))


def hash_layer(input: torch.Tensor) -> torch.Tensor:
    """
    Apply the custom hash function.
    """
    return HashFunction.apply(input) 

def cont_layer(input: torch.Tensor) -> torch.Tensor:
    """
    Apply the continuous layer function.
    """
    return ContinuousFunction()(input)
