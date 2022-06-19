import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from model_Transformer import LSTM
from model_FCNN import FCNN
from myLoss import MylossFun
from dataset import MarathonDataset
from sklearn.model_selection import StratifiedKFold
import argparse

#======Arguement setting======#
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="choose model (Transformer or FCNN)")
args = parser.parse_args()

#======GPU setting======#
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#======Load trainset======#
traindataset = MarathonDataset(type='train',model=args.model)
splits=StratifiedKFold(n_splits=5,shuffle=True, random_state=0)
Label = traindataset.get_label()

batch_loss = []
epoch_ = []
loss_epoch = [0]*100
#=====Cross Validation=====#
for fold, (train_idx,val_idx) in enumerate(splits.split(Label,Label)):
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    trainloader = DataLoader(traindataset, batch_size=64, sampler=train_sampler)
    validloader = DataLoader(traindataset, batch_size=64, sampler=valid_sampler)
    
    if args.model == 'Transformer':
        Model = LSTM().to(device)
        LR = 0.0002
    elif args.model == 'FCNN':
        Model = FCNN().to(device)
        LR = 0.002

    loss_ = MylossFun()
    optimizer = optim.Adam(Model.parameters(), lr = LR)

    epoch_loss = []
    valid_loss = []
    accuracy = []
    accuracy_ = []
    ep = 0

    LOSS = float('inf')
    for epoch in range(0,70):
        running_loss = 0.0
        total_loss = 0.0
        Model.train()

        #======Training======#
        for i,(index,x,y) in enumerate(trainloader):
            optimizer.zero_grad()

            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)

            y_pred = Model(x)
            y_pred = y_pred.squeeze()

            l = loss_(y_pred,y)
            l.backward()
            optimizer.step()

            total_loss += l
            n=i
            
        epoch_loss.append(total_loss.data.cpu().numpy()/n)
        result = []
        Model.eval()
        
        #======Calculate trainset accuracy======#
        for i,(index,x,y) in enumerate(trainloader):
            x = torch.Tensor(x).to(device)

            y_pred = Model(x)
            y_pred = y_pred.squeeze()
            y_pred = y_pred.data.cpu().numpy()

            for j,batch in enumerate(y_pred):
                y_ = y[j,:]
                if np.argmax(y_) == np.argmax(batch):
                    result.append(1)
                else:
                    result.append(0)
        accuracy.append(np.mean(result))

        total_loss = 0
        #======Calculate validset accuracy and loss======#
        with torch.no_grad():
            for i,(index,x,y) in enumerate(validloader):
                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y).to(device)

                y_pred = Model(x)
                y_pred = y_pred.squeeze()
                l = loss_(y_pred,y)
                total_loss += l

                y_pred = y_pred.data.cpu().numpy()
                y = y.data.cpu().numpy()
                n=i

                for j,batch in enumerate(y_pred):
                    y_ = y[j,:]
                    if np.argmax(y_) == np.argmax(batch):
                        result.append(1)
                    else:
                        result.append(0)
            accuracy_.append(np.mean(result))
            valid_loss.append(total_loss.data.cpu().numpy()/n)
            print('Validation loss / Batch%2d/Epoch%2d: %05f' %(fold,epoch,total_loss/n))

            if total_loss/n<LOSS:
                LOSS = total_loss/n
                ep = epoch
                print('------------Model updated!--------')
            if epoch == 31:
                path = 'model/' + args.model + '.pt'
                torch.save(Model.state_dict(), path)

        plt.figure()
        plt.plot(accuracy,color = 'r',label = 'training accuracy')
        plt.plot(accuracy_, color = 'g', label = 'validation accuracy')
        plt.legend(loc='upper left')
        plt.grid(linestyle='-.')
        plt.title('Accuracy vs Epochs, Fold_{}'.format(fold))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig("ACC/ACC_{}.png".format(fold), dpi=300, format = 'png')

        plt.figure()
        plt.plot(epoch_loss,color = 'r',label = 'training loss')
        plt.plot(valid_loss, color = 'g', label = 'validation loss')
        plt.legend(loc='upper right')
        plt.grid(linestyle='-.')
        plt.title('Loss vs Epochs')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("LOSS/LOSS_{}.png".format(fold), dpi=300, format = 'png')\

    loss_epoch = [a + b for a, b in zip(loss_epoch, valid_loss)]
    batch_loss.append(LOSS.data.cpu().numpy())
    epoch_.append(ep)
    print('\n\n')
    print('Batch loss: {}'.format(list(batch_loss)) )
    print('Correspoding Epoch: {}'.format(epoch_) )
    print('Avg loss: %5f' %np.mean(batch_loss))
    print('Avg epoch: %2f' %np.mean(epoch_))
    print('\n\n')

loss_epoch = np.asarray(loss_epoch)/5
plt.figure()
plt.plot(loss_epoch)
plt.grid(linestyle='-.')
plt.title('Loss vs Epochs, Average')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("AvgLOSS.png", dpi=300, format = 'png')

print('Avg loss: %5f' %np.min(loss_epoch))
print('Avg epoch: %5f' %np.argmin(loss_epoch))
