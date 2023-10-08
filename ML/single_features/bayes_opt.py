import copy
import pickle
from functools import partial
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from typing import Tuple
import random


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


def fn(space, x, y):
    model = SVC(**space)
    model.fit(x, y)
    prob_y = model.predict_proba(x)[:, 1]
    score = roc_auc_score(y, prob_y)
    return score


def ml_fn(space, x, y, model, cv=None, random_state=None):
    kF = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    batch_metrics = []
    model_ = copy.deepcopy(model).set_params(**space)
    for idx, (index_train, index_test) in enumerate(kF.split(x)):
        # print("CAT", idx)
        x_tr, y_tr = x.iloc[index_train, :], y[index_train]
        x_te, y_te = x.iloc[index_test, :], y[index_test]
        model_.fit(x_tr, y_tr)
        prob_y = model_.predict_proba(x_te)[:, 1]
        score = roc_auc_score(y_te, prob_y)
        batch_metrics.append(score)
    return - round(sum(batch_metrics) / len(batch_metrics), 8)
    # return max(batch_metrics)


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


def dl_search(x, y, model_params, model_name, des_name, end_p_name, save_tpe=True, return_tpe=False,
              verbose=False, loss_print=True):
    T = TrailRecord()
    tpe_params = dict()
    for key, values in model_params.items():
        tpe_params.update({key: hp.choice(key, values)})
    fn_ = partial(fn, x=x, y=y)
    esf_ = partial(esf, verbose=loss_print, params=model_params, t=T)
    best_results = fmin(fn_, tpe_params, tpe.suggest, max_evals=50, return_argmin=False, trials=Tr, early_stop_fn=esf_,
                        verbose=verbose)
    print()
    if save_tpe:
        with open("bayes_ree/{}_{}_{}_bayes_results.pkl".format(end_p_name, des_name, model_name), "wb") as f:
            pickle.dump(T, f)

    if not return_tpe:
        return best_results
    else:
        return best_results, T


class BayesOptimizer(object):
    def __init__(self, model=None, params=None, cv=None, random_state=None, max_iter=None, print_loss=False,
                 verbose=False, early_stop_iter=30, record_t=None):
        self.model = copy.deepcopy(model)
        self.params = params
        self.cv = cv
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
        self.T = copy.deepcopy(record_t)
        self.Tr = Trials()

    @staticmethod
    def _input_correct(x, y) -> Tuple[pd.DataFrame, np.ndarray]:
        assert isinstance(x, (np.ndarray, pd.DataFrame)), "Datatype x is error!"
        assert isinstance(y, (np.ndarray, pd.DataFrame)), "Datatype y is error!"

        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=None)
        if isinstance(y, pd.DataFrame):
            y = np.array(y)
        return x, y

    def fit(self, x, y):
        x, y = self._input_correct(x, y)
        tpe_params = dict()
        for key, values in self.params.items():
            tpe_params.update({key: hp.choice(key, values)})
        fn_ = partial(ml_fn, x=x, y=y, model=copy.deepcopy(self.model), cv=self.cv, random_state=self.random_state)
        esf_ = partial(esf, t=self.T, verbose=self.loss_print, params=self.params, early_stop_iter=self.early_stop_iter)
        self.best_params_ = self.f_min(fn_, tpe_params, tpe.suggest,
                                       max_evals=self.max_iter,
                                       return_argmin=False,
                                       trials=self.Tr,
                                       early_stop_fn=esf_,
                                       verbose=self.verbose,
                                       )
        print()
        self.best_estimator_ = self.model.set_params(**self.best_params_)

    def results_saving(self, end_p_name, des_name, model_name):
        with open("bayes_results_save/{}_{}_{}_bayes_results.pkl".format(end_p_name, des_name, model_name), "wb") as f:
            pickle.dump(self.T, f)


if __name__ == "__main__":
    pass
