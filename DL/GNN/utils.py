from tqdm import tqdm
import dgl
import torch
from dgl.dataloading import GraphDataLoader as GDl
from dgl.data import DGLDataset
from abc import ABC
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import pandas as pd
from typing import Union
from typing import Tuple


class PYG2DGL(DGLDataset):
    def __init__(self, load_data):
        super().__init__(name="cyp")
        self.load_data = load_data
        self.y = []
        self.smiles = []
        self.smiles_size = []
        self.graph = []
        self.load_pyg()

    def load_pyg(self):
        for i in tqdm(self.load_data):
            g = dgl.graph((i.edge_index[0], i.edge_index[1]))
            g.ndata["features"] = i.x
            edge_features = torch.concat([i.edge_feature[int(e)-1, :].unsqueeze(0) for e in i.edge_index[0]], dim=0)
            g.edata["features"] = edge_features
            self.y.append(i.y)
            self.smiles += [i.smiles]
            self.smiles_size.append(int(i.smiles_size))
            self.graph.append(g)

    def process(self):
        pass

    def __getitem__(self, item):
        return self.graph[item], self.y[item], self.smiles[item], self.smiles_size[item]

    def __len__(self):
        return len(self.smiles)


class DataCollate(InMemoryDataset, ABC):
    def __init__(self, load_data):
        super().__init__()
        self.data, self.slices = load_data


def data_collate(path):
    load_data = torch.load(path)
    data_c = DataCollate(load_data=load_data)
    return data_c


def split_train_valid(train_data ) -> Tuple[list, list]:
    y = [int(data.y) for data in train_data]
    train_x, valid_x = train_test_split(train_data,
                                        random_state=100, shuffle=True, train_size=0.9,
                                        stratify=y)
    return train_x, valid_x


def data_pipline(train_path, valid_path, test_path, method="PYG"):
    assert method == "PYG" or method == "DGL", "Method error: ---> PYG/DGL but input {}".format(method)

    train_data = data_collate(train_path)
    test_data = data_collate(test_path)
    valid_data = data_collate(valid_path)
    # train_data, valid_data = split_train_valid(train_data)
    inm = InMemoryDataset()

    y_ = [int(i.y) for i in train_data]
    training_weight = [len(y_) / (len(y_) - sum(y_)), len(y_) / sum(y_)]

    if method == "PYG":
        train_data = DataCollate(inm.collate(train_data))
        valid_data = DataCollate(inm.collate(valid_data))

        train_data = DataLoader(dataset=train_data, batch_size=256,
                                shuffle=True, drop_last=False)
        valid_data = DataLoader(dataset=valid_data, batch_size=128,
                                shuffle=False, drop_last=False)
        test_data = DataLoader(dataset=test_data, batch_size=128,
                               shuffle=False, drop_last=False)
        return train_data, test_data, valid_data, training_weight

    else:
        train_data = PYG2DGL(DataCollate(inm.collate(train_data)))
        valid_data = PYG2DGL(DataCollate(inm.collate(valid_data)))
        test_data = PYG2DGL(test_data)

        train_data = GDl(dataset=train_data, batch_size=256,
                      shuffle=True, drop_last=False)
        valid_data = GDl(dataset=valid_data, batch_size=128,
                      shuffle=False, drop_last=False)
        test_data = GDl(dataset=test_data, batch_size=128,
                      shuffle=False, drop_last=False)
        return train_data, test_data, valid_data, training_weight

def data_pipline_for_test(test_path, method="PYG"):
    assert method == "PYG" or method == "DGL", "Method error: ---> PYG/DGL but input {}".format(method)
    test_data = data_collate(test_path)

    # train_data, valid_data = split_train_valid(train_data)
    inm = InMemoryDataset()

    if method == "PYG":
        test_data = DataLoader(dataset=test_data, batch_size=128,
                                shuffle=False, drop_last=False)
        return test_data

    else:
        test_data = PYG2DGL(test_data)
        test_data = GDl(dataset=test_data, batch_size=128,
                      shuffle=False, drop_last=False)
        return test_data

def model_init(model, init_method="xavier"):
    for name, w in model.named_parameters():
        if 'weight' in name:
            if len(w.shape) >= 2:
                if init_method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            else:
                nn.init.normal_(w)
        elif 'bias' in name:
            nn.init.constant_(w, 0)
    return model


def results_combine(model_results, columns_name, epoch_, model_loss, model_name=None, data_class=None):
    results_save = pd.DataFrame([model_results], columns=columns_name)
    if model_name is None:
        loss_combine = pd.DataFrame([[epoch_, model_loss]], columns=['epoch', 'loss'])
    else:
        loss_combine = pd.DataFrame([[model_name, data_class, epoch_, model_loss]], columns=['model', "class", 'epoch', 'loss'])
    results_f = pd.concat([loss_combine, results_save], axis=1)
    return results_f
