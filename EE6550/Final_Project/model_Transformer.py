import torch.nn as nn
import torch
import torch.nn.functional as F

class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=17, nhead=17)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
    
    def forward(self, input):
        input = input.permute(1,0,2)
        output = self.encoder(input)
        output = output.permute(1,0,2)
        return output

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(17, 48, num_layers=2,bidirectional=True)
        self.embedding = nn.Linear(96, 48)

        cnn = nn.Sequential()
        cnn.add_module('Conv1',nn.Conv1d(48, 36, kernel_size = 3, padding = 1))
        cnn.add_module('Batch1', nn.BatchNorm1d(36))
        cnn.add_module('ReLu1',nn.PReLU())
        cnn.add_module('Conv3',nn.Conv1d(36, 24, kernel_size = 3, padding = 1))
        cnn.add_module('Batch3', nn.BatchNorm1d(24))
        cnn.add_module('ReLu3',nn.PReLU())
        cnn.add_module('Conv4',nn.Conv1d(24, 12, kernel_size = 3, padding = 1))
        cnn.add_module('Batch4', nn.BatchNorm1d(12))
        cnn.add_module('ReLu4',nn.PReLU())
        self.cnn = cnn

        cnn2 = nn.Sequential()
        cnn2.add_module('Conv1',nn.Conv1d(29, 1, kernel_size=1))
        cnn2.add_module('Linear1',nn.Linear(14, 5))
        self.cnn2 = cnn2

        self.softmax = nn.Softmax(dim=2)
        self.transformer = transformer()

    def forward(self, input):
        #=====doing one hot encoding for input=====#
        input_ = (input[:,:,0]-1).to(torch.int64)
        input_ = F.one_hot(input_, num_classes=14).to(torch.float)
        input = torch.cat((input_, input[:,:,1:4]),2)
        #=====applying transformer=====#
        input1 = self.transformer(input)      
        input = input1+input
        input = input.permute(0,2,1)
        
        #=====applying CNN uNet structure with LSTM=====#
        input_ = input.permute(2,0,1)
        recurrent, _ = self.rnn(input_)       
        T, b, h = recurrent.size()
        output = self.embedding(recurrent)
        output = output.permute(1,2,0)
        
        output = self.cnn(output)
        output = torch.cat((output,input),1)
        output = self.cnn2(output)
        output = self.softmax(output)
        return output