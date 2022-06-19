import torch.nn as nn
import torch.nn.functional as F

class FCNN(nn.Module):

    def __init__(self):
        super(FCNN, self).__init__()
        self.linear = nn.Sequential(nn.Linear(17,32),
                                     nn.PReLU(),
                                     nn.Linear(32,16),
                                     nn.PReLU(),
                                     nn.Linear(16,5),
                                     nn.PReLU())
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        input = F.normalize(input)
        output = self.linear(input)
        output = self.softmax(output)            
        return output

