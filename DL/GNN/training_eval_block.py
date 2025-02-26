import torch
from model_params import *
import numpy as np
from utils import *
from public_metrics import *


def train(model, data, optimizer, loss_function, weight, attn=True, verbose=False):
    for index, data_i in enumerate(data):
        if isinstance(data_i, list):
            # DGL data == list   PYG == Data
            data_i = [data_i[i].to(devices) for i in range(2)]
            y = data_i[1]
        else:
            data_i = data_i.to(devices)
            y = data_i.y
        model.train()
        if attn:
            output, _ = model(data_i)
        else:
            output = model(data_i)
        batch_weight = torch.tensor([weight[0] if y_ == 0 else weight[1] for y_ in y]).to(devices)
        if isinstance(data_i, list):
            ls = loss_function(output, y.long().reshape(-1))
        else:
            ls = loss_function(output, y)
        ls = torch.mean(ls * batch_weight)
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        if verbose is True:
            print("train loss:", ls.item())


def eval_(model, data, loss_function, weight, attn=True, output_loss=True, eval_class=None, verbose=True):
    batch_loss, batch_y, batch_prob = [], [], []
    if eval_class == "train":
        n = 3
    else:
        n = 1
    for index, data_i in enumerate(data):
        if isinstance(data_i, list):
            data_i = [data_i[i].to(devices) for i in range(2)]
            y = data_i[1]
        else:
            data_i = data_i.to(devices)
            y = data_i.y
        model.eval()
        with torch.no_grad():
            if index % n == 0:
                if attn is True:
                    eval_output, _ = model(data_i)
                else:
                    eval_output = model(data_i)

                if output_loss:
                    b_weight = torch.tensor([weight[0] if y_ == 0 else weight[1] for y_ in y]).to(devices)
                    if isinstance(data_i, list):
                        ls = loss_function(eval_output, y.long().reshape(-1))
                    else:
                        ls = loss_function(eval_output, y)
                    ls = torch.mean(ls * b_weight)
                    batch_y += y.reshape(-1).data.cpu()
                    batch_prob += eval_output[:, 1].reshape(-1).data.cpu()
                    batch_loss.append(ls.item())

                    # 传出三个值 损失值， batch_y, eval_y
                else:
                    batch_y += y.reshape(-1).data.cpu()
                    batch_prob += eval_output[:, 1].reshape(-1).data.cpu()
                    batch_loss.append(None)

    r, c = metric(np.array(batch_y), np.array(batch_prob), verbose=verbose)
    try:
        return r, c, np.mean(batch_loss)
    except:
        return r, c, []

