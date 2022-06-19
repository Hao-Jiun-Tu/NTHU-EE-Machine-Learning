import torch
from torch import det, detach
import torch.nn as nn
from torch.autograd import Variable

class MylossFun(nn.Module):
    def __init__(self, ):
        super(MylossFun, self).__init__()

    def forward(self, out, target):
        b,l = target.size()

        out = torch.log(out)
        loss = out*target
        loss = torch.sum(loss)
        loss = -loss/b
        return loss