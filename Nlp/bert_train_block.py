from utils import *


def train_step(data, model, loss_func1, optimizer, auc_block, loss_block):
    model.train()
    for x, properties in data:
        clf_true = properties['clf']
        properties_pred = model(x)
        clf_pred = properties_pred['clf']

        loss = 0
        # 仅计算有效的loss的总和/有效loss的个数（求均值）
        loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (clf_true != -1000).float()).sum() / (
                (clf_true != -1000).float().sum() + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        auc_block.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
        loss_block.update(loss.detach().cpu().item(), x.shape[0])


def eval_step(data, model, loss_func1, optimizer, auc_block, loss_block):
    model.eval()
    with torch.no_grad():
        for x, properties in data:
            clf_true = properties['clf']
            properties_pred = model(x)
            clf_pred = properties_pred['clf']
            loss = 0
            loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                    clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

            auc_block.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())

            loss_block.update(loss.detach().cpu().item(), x.shape[0])



train_results_save, valid_results_save, test_results_save = None, None, None