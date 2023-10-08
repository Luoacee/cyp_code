import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import  BertModel
from dataset import Smiles_Bert_Dataset,Pretrain_Collater
import time
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from metrics import AverageMeter
import argparse

main_seed = 100
random.seed(main_seed)
torch.manual_seed(main_seed)
torch.cuda.manual_seed(main_seed)
np.random.seed(main_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--Smiles_head', nargs='+', default=["Smiles"], type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型参数
small = {'name': 'small', 'num_layers': 4, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights'}
medium = {'name': 'medium', 'num_layers': 6, 'num_heads': 6, 'd_model': 300, 'path': 'medium_weights'}
large = {'name': 'large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights'}

arch = medium    ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']

dff = d_model*4
vocab_size = 60
dropout_rate = 0.1

model = BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)
params_n = 0
for ps in model.parameters():
    if ps.requires_grad == True:
        params_n += 1
print("params_number:", params_n)


model.to(device)

# data = pd.read_csv('data/chem.csv')

full_dataset = Smiles_Bert_Dataset('data/Valid_smiles_32.csv',Smiles_head=args.Smiles_head)

# 划分很小的数据集
train_size = int(0.98 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
# => x, y, weight
train_dataloader = DataLoader(train_dataset,batch_size=512,shuffle=True,collate_fn=Pretrain_Collater())
test_dataloader = DataLoader(test_dataset,batch_size=512,shuffle=False,collate_fn=Pretrain_Collater())

# 模型参数设定
optimizer = optim.Adam(model.parameters(),1e-4,betas=(0.9,0.98))
# pad_value = 0
loss_func = nn.CrossEntropyLoss(ignore_index=0,reduction='none')

train_loss = AverageMeter()
train_acc = AverageMeter()
test_loss = AverageMeter()
test_acc = AverageMeter()

def train_step(x, y, weights):
    model.train()
    optimizer.zero_grad()
    predictions = model(x)

    # 仅有mask进行了贡献
    loss = (loss_func(predictions.transpose(1,2),y)*weights).sum()/weights.sum()
    loss.backward()
    optimizer.step()

    train_loss.update(loss.detach().cpu().item(),x.shape[0])
    train_acc.update(((y==predictions.argmax(-1))*weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                     weights.cpu().sum().item())


def test_step(x,y, weights):
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum()/weights.sum()

        test_loss.update(loss.detach().cpu().item(), x.shape[0])
        test_acc.update(((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                              weights.cpu().sum().item())

train_epoch_acc, train_epoch_loss = [], []
test_epoch_acc, test_epoch_loss = [], []
for epoch in range(2):
    start = time.time()

    train_batch_acc, train_batch_loss = [], []
    test_batch_acc, test_batch_loss = [], []
    for (batch, (x, y, weights)) in enumerate(train_dataloader):
        train_step(x, y, weights)

        if batch%500==0:
            print('Epoch {} Batch {} training Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.avg))
            print('traning Accuracy: {:.4f}'.format(train_acc.avg))
            train_batch_acc.append(train_acc.avg)
            train_batch_loss.append(train_loss.avg)

        if batch % 1000 == 0:
            for x, y ,weights in test_dataloader:
                test_step(x, y , weights)
            print('Test loss: {:.4f}'.format(test_loss.avg))
            print('Test Accuracy: {:.4f}'.format(test_acc.avg))

            test_batch_acc.append(test_acc.avg)
            test_batch_loss.append(test_loss.avg)
        test_acc.reset()
        test_loss.reset()
        train_acc.reset()
        train_loss.reset()
    train_epoch_loss.append(np.mean(train_batch_loss)), train_epoch_acc.append(np.mean(train_batch_acc))
    test_epoch_loss.append(np.mean(test_batch_loss)), test_epoch_acc.append(np.mean(test_batch_acc))
    print('Epoch {} is Done!'.format(epoch))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, train_loss.avg))
    print('training Accuracy: {:.4f}'.format(train_acc.avg))
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.avg))
    print('test Accuracy: {:.4f}'.format(test_acc.avg))
    torch.save(model.state_dict(),'weights/' + arch['path']+'_bert_weights{}_{}.pt'.format(arch['name'],epoch+1) )
    torch.save(model.encoder.state_dict(), 'weights/' + arch['path'] + '_bert_encoder_weights{}_{}.pt'.format(arch['name'], epoch + 1))

full_pd = pd.DataFrame(np.array([train_epoch_acc, train_epoch_loss, test_epoch_acc, test_epoch_loss]).T, columns=["train_acc",
                                                                                                      "train_loss",
                                                                                                      "test_acc",
                                                                                                      "test_loss"])
full_pd.to_csv("results.csv", index=False)