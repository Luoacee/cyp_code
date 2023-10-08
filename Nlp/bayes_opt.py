import copy
import random

from hyperopt import fmin, tpe, hp, Trials
from torch import optim
from main import *


class TrailNode:
    def __init__(self, tid, loss, params):
        self.tid = tid
        self.loss = loss
        self.params = params

    def __call__(self):
        return dict(tid=self.tid, loss=self.loss, params=self.params)

    def __repr__(self):
        return "{}(tid: {})".format(self.__class__.__name__, self.tid)


class TrailRecord:
    def __init__(self):
        self.record = list()
        self.len = None
        self.min_loss = None

    def get_min_loss(self):
        loss_ = [r.loss for r in self.record]
        return min(loss_)

    def __call__(self, node):
        self.record.append(node)

    def __len__(self):
        self.len = len(self.record)
        return self.len

    def __repr__(self):
        self.__len__()
        return "{}({})".format(self.__class__.__name__, self.len)


# epoch
def _fn_epoch(space, model, model_name, train_df, valid_df, epochs_, weight, attn, device):
    print("Get params: {}".format(str(space)))

    model_p = {k: v for k, v in space.items() if k != "lr"}
    model_ = model(**model_p).to(device)
    model_ = model_init(model_)
    # optimizer ->
    optimizer = optim.AdamW(model_.parameters(),
                            space["lr"],
                            weight_decay=0.001)
    model_valid_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    params = sum([np.prod(p.size()) for p in model_valid_parameters])
    loss_function = nn.CrossEntropyLoss(reduction="none")

    if epochs_ >= 15:
        auc_sample1 = [epochs_ - 10, epochs_ - 7, epochs_ - 4, epochs_ - 1]
        auc_sample2 = [epochs_ - 11, epochs_ - 8, epochs_ - 5, epochs_ - 2]
        auc_sample3 = [epochs_ - 12, epochs_ - 9, epochs_ - 6, epochs_ - 3]
        auc_m = [[], [], []]

        valid_results_save, train_results_save = None, None
        for epoch in range(epochs_):
            train(model_, train_df, optimizer, loss_function, weight, attn)
            valid_results, valid_columns, valid_loss = eval_(model_, valid_df, loss_function, weight,
                                                             attn, verbose=True)
            train_results, train_columns, train_loss = eval_(model_, train_df, loss_function,
                                                             weight, attn, eval_class="train", verbose=False)
            auc = valid_results[0]

            if epoch == 0:
                valid_results_save = results_combine(valid_results, valid_columns, epoch, valid_loss)
                train_results_save = results_combine(train_results, train_columns, epoch, train_loss)
            else:
                valid_results_save = pd.concat([valid_results_save,
                                                results_combine(valid_results, valid_columns, epoch, valid_loss)],
                                               axis=0)
                train_results_save = pd.concat([train_results_save,
                                                results_combine(train_results, train_columns, epoch, train_loss)],
                                               axis=0)
            if epoch in auc_sample1:
                auc_m[0].append(auc)
            elif epoch in auc_sample2:
                auc_m[1].append(auc)
            elif epoch in auc_sample3:
                auc_m[2].append(auc)
            else:
                pass
            try:
                [pds.to_csv("opt_results/{}/{}/H_{}_O_{}_D_{}_lr_{}_{}.csv".format(
                    model_name[0],
                    model_name[1],
                    space["hidden_dim"],
                    space["out_dim"],
                    space["head_n"],
                    space["lr"],
                    pds_name
                )) for pds_name, pds in zip(("train", "valid"), (train_results_save, valid_results_save))]
            except KeyError:
                [pds.to_csv("opt_results/{}/{}/H_{}_O_{}_lr_{}_{}.csv".format(
                    model_name[0],
                    model_name[1],
                    space["hidden_dim"],
                    space["out_dim"],
                    space["lr"],
                    pds_name
                )) for pds_name, pds in zip(("train", "valid"), (train_results_save, valid_results_save))]

        auc_mean = np.mean(np.mean(auc_m[0]) + np.mean(auc_m[1]) + np.mean(auc_m[2]))
        return -auc_mean
    else:
        "For coding test!"
        auc_record = []
        for epoch in range(epochs_):
            train(model_, train_df, optimizer, loss_function, weight, attn)
            valid_results, _, _ = eval_(model_, valid_df, loss_function, weight, attn)
            _, _, _ = eval_(model_, train_df, loss_function, weight, attn)
            auc_record.append(valid_results[0])
        if len(auc_record) >= 4:
            return -np.mean(auc_record[-4:])
        else:
            return -np.mean(auc_record)


def _bfn(space, model, model_name, train_df, valid_df, epochs_, weight, device):
    print("Get params: {}".format(str(space)))
    model_p = {k: v for k, v in space.items() if k != "lr" and k != "pt_epoch"}
    model_ = model(**model_p).to(device)
    model_.encoder.load_state_dict(torch.load(
        'pt_model_save/MTL_BERT/medium_weights_bert_encoder_weightsmedium_{}.pt'.format(space["pt_epoch"])))
    # optimizer ->
    optimizer = torch.optim.AdamW(model_.parameters(), lr=space["lr"], betas=(0.9, 0.98))
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

    train_loss = AverageMeter()
    valid_loss = AverageMeter()

    train_aucs = Metrics()
    valid_aucs = Metrics()


    valid_results_save, train_results_save = None, None
    best_metrics = 0
    state_supervised = 0
    for epoch in range(epochs_):
        train_step(train_df, model_, loss_function, optimizer, train_aucs, train_loss)

        # print('epoch: ', epoch, 'train loss: {:.4f}'.format(train_loss.avg))
        train_results, train_columns = train_aucs.results()
        tr_loss = train_loss.avg
        eval_step(valid_df, model_, loss_function, optimizer, valid_aucs, valid_loss)
        # print('epoch: ', epoch, 'valid loss: {:.4f}'.format(valid_loss.avg))
        valid_results, valid_columns = valid_aucs.results()
        va_loss = valid_loss.avg

        epoch_metrics = valid_results[0]
        if epoch_metrics > best_metrics:
            best_metrics = epoch_metrics
            state_supervised = 0
        else:
            state_supervised += 1
        if state_supervised == 30:
            break

        if epoch == 0:
            valid_results_save = results_combine(valid_results, valid_columns, epoch, va_loss)
            train_results_save = results_combine(train_results, train_columns, epoch, tr_loss)
        else:
            valid_results_save = pd.concat([valid_results_save,
                                            results_combine(valid_results, valid_columns, epoch, va_loss)],
                                           axis=0)
            train_results_save = pd.concat([train_results_save,
                                            results_combine(train_results, train_columns, epoch, tr_loss)],
                                           axis=0)
        valid_aucs.reset()
        valid_loss.reset()
        train_aucs.reset()
        train_loss.reset()


    [pds.to_csv("opt_results/{}/{}/D_{}_lr_{}_{}.csv".format(
        model_name[0],
        model_name[1],
        space["dropout_rate"],
        space["lr"],
        pds_name
    )) for pds_name, pds in zip(("train", "valid"), (train_results_save, valid_results_save))]
    return -best_metrics

def _fn(space, model, model_name, train_df, valid_df, epochs_, weight, device):
    print("Get params: {}".format(str(space)))
    model_p = {k: v for k, v in space.items() if k != "lr"}
    print(model_p)
    model_ = model(**model_p).to(device)
    model_ = model_init(model_)
    model_.public_model.load_state_dict(torch.load("pt_model_save/{}/epoch_9.pth".format(model_name[1])))
    # optimizer ->
    optimizer = optim.Adam(model_.parameters(),
                           space["lr"])
    loss_function = nn.CrossEntropyLoss(reduction="none")

    valid_results_save, train_results_save = None, None
    best_metrics = 0
    state_supervised = 0
    for epoch in range(epochs_):
        train(model_, train_df, optimizer, loss_function, weight)
        train_results, train_columns, train_loss = eval_(model_, train_df, loss_function,
                                                         weight, eval_class="train", verbose=False)
        valid_results, valid_columns, valid_loss = eval_(model_, valid_df, loss_function, weight,
                                                         verbose=False)

        epoch_metrics = valid_results[0]
        if epoch_metrics > best_metrics:
            best_metrics = epoch_metrics
            state_supervised = 0
        else:
            state_supervised += 1
        if state_supervised == 30:
            break

        if epoch == 0:
            valid_results_save = results_combine(valid_results, valid_columns, epoch, valid_loss)
            train_results_save = results_combine(train_results, train_columns, epoch, train_loss)
        else:
            valid_results_save = pd.concat([valid_results_save,
                                            results_combine(valid_results, valid_columns, epoch, valid_loss)],
                                           axis=0)
            train_results_save = pd.concat([train_results_save,
                                            results_combine(train_results, train_columns, epoch, train_loss)],
                                           axis=0)

    try:
        [pds.to_csv("opt_results/{}/{}/H_{}_I_{}_M_{}_lr_{}_{}.csv".format(
            model_name[0],
            model_name[1],
            space["hidden_dim"],
            space["input_dim"],
            space["model_dim"],
            space["lr"],
            pds_name
        )) for pds_name, pds in zip(("train", "valid"), (train_results_save, valid_results_save))]
    except KeyError:
        [pds.to_csv("opt_results/{}/{}/H_{}_I_{}_lr_{}_head_{}_K_{}_{}.csv".format(
            model_name[0],
            model_name[1],
            space["hidden_dim"],
            space["input_dim"],
            space["lr"],
            space["head"],
            space["kernel_size"],
            pds_name
        )) for pds_name, pds in zip(("train", "valid"), (train_results_save, valid_results_save))]

    return -best_metrics


def esf(*args, params, verbose=True, t=None, early_stop_iter=None):
    final_i = None

    for i in args[0]:
        final_i = i
    params_ = dict()
    for key_, values_ in params.items():
        params_.update({key_: values_[final_i["misc"]["vals"][key_][0]]})
    node = TrailNode(tid=final_i["tid"], loss=final_i["result"]["loss"],
                     params=params_)
    t(node)
    if verbose:
        print("\rIter: {}   {}  Min loss is :{}".format(final_i["tid"], ["->->->->-"
                                                                         if final_i["tid"] % 2 == 0
                                                                         else ">->->->->"][0], t.get_min_loss()),
              end="")
    else:
        pass

    try:
        mark = args[1]
    except IndexError:
        mark = 0
        pass

    if mark == early_stop_iter:
        print("\nEarly stop: ", end="")
        return True, []

    try:
        if t.min_loss == t.get_min_loss():
            mark += 1
        else:
            mark = 0
    except TypeError:
        pass

    t.min_loss = t.get_min_loss()
    return False, [mark]


class BayesOptimizer(object):
    def __init__(self, model=None, model_name=None, params=None, epoch=None, random_state=None, max_iter=None,
                 identify_params = None,
                 print_loss=False,
                 verbose=False, early_stop_iter=30, weight=None, device=None):
        self.model = model
        self.params = params
        self.epoch = epoch
        self.loss_print = print_loss
        self.verbose = verbose
        self.max_iter = max_iter
        self.early_stop_iter = early_stop_iter
        self.f_min = copy.deepcopy(fmin)
        if not random_state:
            self.random_state = random.random()
        else:
            self.random_state = random_state
        self.best_params_ = None
        self.best_estimator_ = None
        self.T = copy.deepcopy(TrailRecord())
        self.Tr = Trials()
        self.device = device
        self.weight = weight
        self.best_optimizer_ = None
        self.model_name = model_name
        self.identify_params = identify_params

    def fit(self, train_df, valid_df):
        tpe_params = dict()
        for key, values in self.params.items():
            tpe_params.update({key: hp.choice(key, values)})
        assert self.model_name[1] in ["MTL_BERT", "Bi_LSTM", "TextCNN"], "Key error"
        if "MTL_BERT" not in self.model_name:
            insert_fn = _fn
        else:
            insert_fn = _bfn
        fn = partial(
            insert_fn,
            model=self.model,
            model_name=self.model_name,
            train_df=train_df,
            valid_df=valid_df,
            epochs_=self.epoch,
            weight=self.weight,
            device=self.device
        )
        esf_ = partial(esf, t=self.T, verbose=self.loss_print, params=self.params, early_stop_iter=self.early_stop_iter)
        self.best_params_ = self.f_min(fn, tpe_params, tpe.suggest,
                                       max_evals=self.max_iter,
                                       return_argmin=False,
                                       trials=self.Tr,
                                       early_stop_fn=esf_,
                                       verbose=self.verbose
                                       )
        print()
        print(self.best_params_)
        print()
        model_p = {k: v for k, v in self.best_params_.items() if k != "lr" and k != "pt_epoch"}
        self.best_estimator_ = self.model(**model_p)
        if "MTL_BERT" not in self.model_name:
            self.best_estimator_ = model_init(self.best_estimator_)
            self.best_estimator_.public_model.load_state_dict(torch.load("pt_model_save/{}/epoch_9.pth".format(
                                                                                                           self.model_name[1])))
        else:
            self.best_estimator_.encoder.load_state_dict(
                torch.load(
                    'pt_model_save/MTL_BERT/medium_weights_bert_encoder_weightsmedium_{}.pt'.format(
                        self.best_params_["pt_epoch"])))
        self.best_optimizer_ = optim.Adam(self.best_estimator_.parameters(),
                                          self.best_params_["lr"])

    def results_saving(self, end_p_name, des_name, model_name):
        with open("opt_results/{}_{}_bayes_results.pkl".format(end_p_name,model_name), "wb") as f:
            pickle.dump(self.T, f)


if __name__ == "__main__":
    pass
