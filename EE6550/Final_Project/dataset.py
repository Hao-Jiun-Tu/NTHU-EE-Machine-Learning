# pylint: disable=invalid-name
from torch.utils.data import Dataset,DataLoader
import joblib
import numpy as np

class MarathonDataset(Dataset):
    'Marathon'
    def __init__(self, type, model):
        self.type = type
        self.model = model
        if model == 'Transformer':
            if type == 'train' or type == 'valid':
                self.x_dict = joblib.load('dataset_class/train_feature.pkl')
                self.y_dict = joblib.load('dataset_class/train_label.pkl')
            else:
                self.x_dict = joblib.load('dataset_class/test_feature.pkl')
        elif model == 'FCNN':
            if type == 'train' or type == 'valid':
                self.x_dict = joblib.load('dataset_day/train_feature.pkl')
                self.y_dict = joblib.load('dataset_day/train_label.pkl')
            else:
                self.x_dict = joblib.load('dataset_day/test_feature.pkl')

    def __getitem__(self,index):
        if self.type == 'train':
            x = np.asarray(self.x_dict[index][1:])
            x = np.float32(x)
            y = np.asarray(self.y_dict[index][1:])
            y = np.float32(y)
            return index, x, y
        else:
            x = np.asarray(self.x_dict[index][1:])
            x = np.float32(x)
            return index, x, self.x_dict[index][0]

    def __len__(self):
        return len(self.x_dict)


    def get_label(self):
        label = []
        for x_ in self.y_dict.values():
            label.append(np.argmax(x_[1:]))
        return label

