from public_metrics import metric
from copy import deepcopy
from params import *


def pretraining_train(model, data, optimizer, loss_function, verbose=False):
    model.train()
    for index, data_i in enumerate(data):
        x, y, w, lth = data_i
        x = deepcopy(y).to(device)
        y = y[:, 1:].to(device)
        w = w[:, 1:].to(device)
        output = model.pretraining(x, lth)
        valid_output = output[:, :-1, :]
        ls = loss_function(valid_output.transpose(-1, -2), y)
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        if verbose is True:
            if index % 200 == 0:
                print("training loss: ", ls.item())


def pretraining_eval(model, data, loss_function, output_loss=True, output_smiles=False, verbose=True):
    model.eval()
    batch_acc, batch_loss = [], []
    for index, data_i in enumerate(data):
        with torch.no_grad():
            x, y, w, lth = data_i
            x = deepcopy(y).to(device)
            y = y[:, 1:].to(device)
            w = w[:, 1:].to(device)
            lth = lth.to(device)
            eval_output = model.pretraining(x, lth)
            valid_output = eval_output[:, :-1, :]
            ls = loss_function(valid_output.transpose(-1, -2), y)
            arg_max = torch.argmax(valid_output, dim=-1)
            label_record = []
            for idx, i in enumerate(arg_max):
                same_label = (i[:lth[idx] - 1].cpu().data.numpy() == y[idx][:lth[idx] - 1].cpu().data.numpy())
                try:
                    label_record += int(same_label)
                except TypeError:
                    label_record += [int(j) for j in same_label]
                    if output_smiles:
                        smiles_ = []
                        if idx == 0 or idx == 1:
                            for j in i:
                                smiles_ += [i2c[j.item()]]
                            smiles_ = "".join(smiles_)
                            print(smiles_[:lth[idx]])

                full_index = [int(k) for k in label_record]
                acc = sum(full_index) / len(full_index)
                batch_acc.append(acc)
                batch_loss.append(ls.item())
    print("batch acc {}, batch loss{}".format(np.mean(batch_acc), np.mean(batch_loss)))
    return np.mean(batch_acc), np.mean(batch_loss)


def train(model, data, optimizer, loss_function, weight, verbose=False):
    model.train()
    for index, data_i in enumerate(data):
        x, y, lt = data_i
        x = x.to(device)
        y = y.to(device)
        lt = lt.to(device)
        output = model.classification(x, lt)
        batch_weight = torch.tensor([weight[0] if y_ == 0 else weight[1] for y_ in y]).to(device)
        ls = loss_function(output, y.long())
        ls = torch.mean(ls * batch_weight)
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        if verbose is True:
            if index % 3 == 0:
                print("train loss:", ls.item())


def eval_(model, data, loss_function, weight, output_loss=True, eval_class=None, verbose=True):
    batch_loss, batch_y, batch_prob = [], [], []
    if eval_class == "train":
        n = 3
    else:
        n = 1
    for index, data_i in enumerate(data):
        x, y, lt = data_i
        x = x.to(device)
        y = y.to(device)
        lt = lt.to(device)
        model.eval()
        with torch.no_grad():
            if index % n == 0:
                eval_output = model.classification(x, lt)
                if output_loss:
                    b_weight = torch.tensor([weight[0] if y_ == 0 else weight[1] for y_ in y]).to(device)
                    ls = loss_function(eval_output, y.long())
                    ls = torch.mean(ls * b_weight)
                    batch_y += y.reshape(-1).data.cpu()
                    batch_prob += eval_output[:, 1].reshape(-1).data.cpu()
                    batch_loss.append(ls.item())

                else:
                    batch_y += y.reshape(-1).data.cpu()
                    batch_prob += eval_output[:, 1].reshape(-1).data.cpu()
                    batch_loss.append(None)

    r, c = metric(np.array(batch_y), np.array(batch_prob), verbose=verbose)

    try:
        return r, c, np.mean(batch_loss)
    except:
        return r, c, []

