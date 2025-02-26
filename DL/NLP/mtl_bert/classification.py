# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import random
from dataset import Prediction_Dataset, Pretrain_Collater, Finetune_Collater
from sklearn.metrics import r2_score,roc_auc_score
from sklearn.model_selection import train_test_split
from metrics import AverageMeter, Records_R2, Records_AUC, Metrics

import os
from model import  PredictionModel,BertModel
import argparse
main_seed = 100
random.seed(main_seed)
torch.manual_seed(main_seed)
torch.cuda.manual_seed(main_seed)
np.random.seed(main_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--smiles-head', nargs='+', default=['Smiles'], type=str)
parser.add_argument('--clf-heads', nargs='+', default=["Y"], type=str)
parser.add_argument('--reg-heads', nargs='+', default=[], type=list)
args = parser.parse_args()


def split_train_valid(train_data):
    smiles, y = train_data.Smiles.values, train_data.Y.values
    train_x, valid_x, train_y, valid_y = train_test_split(list(smiles), list(y),
                                        random_state=100, shuffle=True, train_size=0.9,
                                        stratify=y)
    return pd.DataFrame(np.array([train_x, train_y]).T, columns=["Smiles", "Y"]), pd.DataFrame(np.array([valid_x, valid_y]).T, columns=["Smiles", "Y"])


def bert_main(seed, cyp_name, ):
    cyp_name = "2c9"
    # small = {'name':'Small','num_layers': 4, 'num_heads': 4, 'd_model': 128,'path':'small_weights'}
    medium = {'name': 'medium', 'num_layers': 6, 'num_heads': 6, 'd_model': 300, 'path': 'medium_weights'}
    # large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 512,'path':'large_weights'}

    arch = medium  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']

    dff = d_model * 4
    vocab_size = 60
    dropout_rate = 0.1

    seed = seed
    np.random.seed(seed=seed)
    # tf.random.set_seed(seed=seed)

    # dfs = []
    # columns = set()
    # for reg_head in args.reg_heads:
    #     df = pd.read_csv('data/reg/{}.csv'.format(reg_head))
    #     # 回归数据zscore归一化
    #     df[reg_head] = (df[reg_head]-df[reg_head].mean())/(df[reg_head].std())
    #     dfs.append(df)
    #     columns.update(df.columns.to_list())
    # for clf_head in args.clf_heads:
    #     df = pd.read_csv('data/clf/{}.csv'.format(clf_head))
    #     dfs.append(df)
    #     columns.update(df.columns.to_list())

    train_data = pd.read_csv("clf/cyp{}_train_MACCS.csv".format(cyp_name)).loc[:, ["Smiles", "Y"]]
    test_data = pd.read_csv("clf/cyp{}_test_MACCS.csv".format(cyp_name)).loc[:, ["Smiles", "Y"]]
    valid_data = pd.read_csv("clf/cyp{}_valid_MACCS.csv".format(cyp_name)).loc[:, ["Smiles", "Y"]]
    columns = ["Smiles", "Y"]
    # train_temps = []
    # test_temps = []
    # valid_temps = []
    # dfs.append(train_data)
    # 8:1:1
    # for df in dfs:
    #     temp = pd.DataFrame(index=range(len(df)), columns=columns)
    #     for column in df.columns:
    #         temp[column] = df[column]
    #     # 全部抽回
    #     temp = temp.sample(frac=1).reset_index(drop=True)
    #     train_temp = temp[:int(0.9*len(temp))]
    #     train_temps.append(train_temp)

    #     # test_temp = temp[int(0.8*len(temp)):int(0.9*len(temp))]
    #     # test_temps.append(test_temp)

    #     valid_temp = temp[int(0.9*len(temp)):]
    #     valid_temps.append(valid_temp)

    # train_df, valid_df = split_train_valid(train_data)
    train_df, valid_df = train_data, valid_data
    test_df = test_data


    train_dataset = Prediction_Dataset(train_df, smiles_head=args.smiles_head,
                                                               reg_heads=args.reg_heads,clf_heads=args.clf_heads)
    test_dataset = Prediction_Dataset(test_df, smiles_head=args.smiles_head,
                                       reg_heads=args.reg_heads, clf_heads=args.clf_heads)
    valid_dataset = Prediction_Dataset(valid_df, smiles_head=args.smiles_head,
                                       reg_heads=args.reg_heads, clf_heads=args.clf_heads)


    train_dataloader = DataLoader(train_dataset, batch_size=256,shuffle=True,collate_fn=Finetune_Collater(args))
    test_dataloader = DataLoader(test_dataset, batch_size=128,shuffle=False,collate_fn=Finetune_Collater(args))
    valid_dataloader = DataLoader(valid_dataset, batch_size=128,shuffle=False,collate_fn=Finetune_Collater(args))


    # x, property = next(iter(train_dataset))
    model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dropout_rate=0.1,reg_nums=len(args.reg_heads),clf_nums=len(args.clf_heads))
    model.encoder.load_state_dict(torch.load(
        '../pt_model_save/MTL_BERT/medium_weights_bert_encoder_weightsmedium_10.pt'))
    model = model.to(device)

    # if pretraining:
    #     model.encoder.load_state_dict(torch.load())
    #     print('load_wieghts')

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.5e-4,betas=(0.9,0.98))
    # lm = lambda x:x/10*(5e-5) if x<10 else (5e-5)*10/x
    # lms = LambdaLR(optimizer,[lm])

    train_loss = AverageMeter()
    test_loss = AverageMeter()
    valid_loss = AverageMeter()

    train_aucs = Metrics()
    test_aucs = Metrics()
    valid_aucs = Metrics()


    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')

    stopping_monitor = 0

    def results_combine(model_results, columns_name, epoch_, model_loss):
        results_save = pd.DataFrame([model_results], columns=columns_name)
        loss_combine = pd.DataFrame([[epoch_, model_loss]], columns=['epoch', 'loss'])
        results_f = pd.concat([loss_combine, results_save], axis=1)
        return results_f

    def train_step(x, properties):
        model.train()
        clf_true = properties['clf']
        properties_pred = model(x)
        clf_pred = properties_pred['clf']

        loss = 0
        # 仅计算有效的loss的总和/有效loss的个数（求均值）
        loss += (loss_func1(clf_pred,clf_true*(clf_true!=-1000).float()) * (clf_true!=-1000).float() ).sum()/((clf_true!=-1000).float().sum()+1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_aucs.update(clf_pred.detach().cpu().numpy(),clf_true.detach().cpu().numpy())

        train_loss.update(loss.detach().cpu().item(),x.shape[0])

    def test_step(x, properties):
        model.eval()
        with torch.no_grad():
            clf_true = properties['clf']
            properties_pred = model(x)

            clf_pred = properties_pred['clf']

            loss = 0

            loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                        clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum()+1e-6)

            test_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())

            test_loss.update(loss.detach().cpu().item(),x.shape[0])

    def valid_step(x, properties):
        model.eval()
        with torch.no_grad():
            clf_true = properties['clf']
            reg_true = properties['reg']
            properties_pred = model(x)

            clf_pred = properties_pred['clf']
            reg_pred = properties_pred['reg']

            loss = 0

            loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                        clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum()+1e-6)

            valid_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())

            valid_loss.update(loss.detach().cpu().item(),x.shape[0])

    train_results_save, valid_results_save, test_results_save = None, None, None

    for epoch in range(200):
        for x,properties in train_dataloader:
            train_step(x,properties)

        print('epoch: ',epoch,'train loss: {:.4f}'.format(train_loss.avg))
        tr_r, tr_c = train_aucs.results()

        for x, properties in valid_dataloader:
            valid_step(x, properties)
        print('epoch: ',epoch,'valid loss: {:.4f}'.format(valid_loss.avg))
        va_r, va_c = valid_aucs.results()

        for x, properties in test_dataloader:
            test_step(x, properties)
        print('epoch: ',epoch,'test loss: {:.4f}'.format(test_loss.avg))
        te_r, te_c = test_aucs.results()

        if epoch == 0:
            valid_results_save = results_combine(va_r, va_c, epoch, np.mean(valid_loss.avg))
            train_results_save = results_combine(tr_r, tr_c, epoch, np.mean(train_loss.avg))
            test_results_save = results_combine(te_r, te_c, epoch, np.mean(test_loss.avg))
        else:
            valid_results_save = pd.concat([valid_results_save,
                                            results_combine(va_r, va_c, epoch, np.mean(valid_loss.avg))], axis=0)
            train_results_save = pd.concat([train_results_save,
                                            results_combine(tr_r, tr_c, epoch, np.mean(train_loss.avg))], axis=0)
            test_results_save = pd.concat([test_results_save,
                                           results_combine(te_r, te_c, epoch, np.mean(test_loss.avg))], axis=0)
        torch.save(model.state_dict(), "ft_model_save/epoch_{}.pth".format(epoch))

        valid_aucs.reset()
        valid_loss.reset()
        train_aucs.reset()
        train_loss.reset()
        test_aucs.reset()
        test_loss.reset()
        print()
    return train_results_save, valid_results_save, test_results_save


if __name__ == '__main__':
    main(100)




