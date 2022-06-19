import csv
import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
from model_Transformer import LSTM
from model_FCNN import FCNN
from dataset import MarathonDataset
import argparse

#======Arguement setting======#
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="choose model (Transformer or FCNN)")
args = parser.parse_args()

#======GPU setting======#
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#======Load model======#
if args.model == 'Transformer':
    Model = LSTM().to(device)
    path = 'model/Transformer.pt'
elif args.model == 'FCNN':
    Model = FCNN().to(device)
    path = 'model/FCNN.pt'
Model.load_state_dict(torch.load(path))
Model.eval()

#======Load testset======#
testdataset = MarathonDataset(type='test',model=args.model)
testloader = DataLoader(dataset=testdataset, batch_size=32, shuffle=False)

result = []
for i,(index,x, mac_hash) in enumerate(testloader):
    mac_hash = np.asanyarray(mac_hash)

    x = torch.Tensor(x).to(device)

    y_pred = Model(x)
    y_pred = y_pred.data.cpu().numpy()

    for j,batch in enumerate(y_pred):
        bb = []
        batch = np.ndarray.tolist(batch)
        bb.append(str(mac_hash[0,j]))
        for b in batch[0]:
            b = str(b)
            bb.append(b)
        result.append(bb)

#======Read submit sample======#
with open('submit_samples.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)

    upload_result = []
    for row1 in rows:
        for row2 in result:
            if row1[0] == row2[0]:
                upload_result.append(row2)

#======Write our predict result======#
with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['mac_hash', 'C0', 'C1', 'C2', 'C3', 'C4'])
    for row in upload_result:
        writer.writerow(row)
