# tools
import collections
import copy
import os
import random
import pickle
import numpy as np
import pandas as pd
from _utils import option_process
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
# metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

# analysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# data
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# model
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from modelParams import rf_params as RF_params
from modelParams import xgboost_params as Xgboost_params
from modelParams import lightgbm_params as LightGBM_params
from modelParams import catboost_params as Catboost_params
from modelParams import svm_params as SVM_params
from _utils2 import DataProcessing
from bayes_opt import BayesOptimizer
from bayes_opt import TrailRecord

random_seed = 100
random.seed(random_seed)

_model = [
    RandomForestClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
    CatBoostClassifier(),
    SVC()
]

subtype = [
    'cyp2c9',
    'cyp2d6',
    'cyp3a4'
]

finger_list = [
    'AD2D',
    'KR',
    'MACCS',
    'MORGAN',
    'MORGAN-F',
    'PUBCHEM',
    'SUBFP',
    'MOLD2',
    'RDKIT'
]

dis_list = [
    'Mold2',
    'D_RDKIT'
]

bayes_n = dict(
    RF=(100, 30),
    Xgboost=(50, 15),
    LightGBM=(100, 30),
    Catboost=(50, 15),
    SVM=(20, 5)
) 

# bayes_n = dict(
#     RF=(5, 30),
#     Xgboost=(20, 5),
#     LightGBM=(5, 30),
#     Catboost=(1, 15),
#     SVM=(20, 5)
# )

_model = dict(
    # RF=_model[0],
    # Xgboost=_model[1],
    # LightGBM=_model[2],
    Catboost=_model[3],
    # SVM=_model[4]
)


def distribute_analysis(x, y, tx, ty):
    def _insert_data_split(_x, _y):
        pos_index = []
        neg_index = []
        _y = _y.values.tolist()
        for idx in range(len(_y)):
            if _y[idx] == 1:
                pos_index.append(idx)
            else:
                neg_index.append(idx)
        return _x.iloc[pos_index, :], _x.iloc[neg_index, :]

    x_pos, x_neg = _insert_data_split(x, y)
    tx_pos, tx_neg = _insert_data_split(tx, ty)
    pca = PCA(n_components=3)
    tsne = TSNE(n_components=3)

    y_pos = [0] * len(x_pos) + [1] * len(tx_pos)
    y_neg = [2] * len(x_neg) + [3] * len(tx_neg)
    x_full_pos = pd.concat([x_pos, tx_pos], axis=0)
    x_full_neg = pd.concat([x_neg, tx_neg], axis=0)
    x_tsne_pos = tsne.fit_transform(x_full_pos)
    x_tsne_neg = tsne.fit_transform(x_full_neg)
    import matplotlib.pylab as plt
    plt.figure(figsize=(6, 4), dpi=300)
    plt.title('POS')
    plt.scatter(x_tsne_pos[:len(x_pos), 0], x_tsne_pos[:len(x_pos), 1], c='#102693', s=2)
    plt.scatter(x_tsne_pos[len(x_pos):, 0], x_tsne_pos[len(x_pos):, 1], c='#DB3520', s=2)
    plt.show()
    plt.figure(figsize=(6, 4), dpi=300)
    plt.title('NEG')
    plt.scatter(x_tsne_neg[:len(x_neg), 0], x_tsne_neg[:len(x_neg), 1], c='#1873AE', s=2)
    plt.scatter(x_tsne_neg[len(x_neg):, 0], x_tsne_neg[len(x_neg):, 1], c='#FF542D', s=2)
    plt.show()


def data_info_split(data, split_method: str = None):
    if split_method == 'train':
        feature_matrix = data.iloc[:, 3:]
        y = data.iloc[:, 2]
        return data, feature_matrix, y

    elif split_method == 'test':
        feature_matrix = data.iloc[:, 2:]
        y = data.iloc[:, 1]
        return data, feature_matrix, y

    else:
        print('Invalid method!')
        _ = ValueError


def train(x, y, t_x, t_y, model_name, sub_name, fg_name):

    print('Bayes optimizer!')
    BO_record = TrailRecord()
    BO = BayesOptimizer(
        copy.deepcopy(_model[model_name]),
        cv=5,
        params=eval('{}_params'.format(model_name)),
        random_state=100,
        print_loss=False,
        verbose=False,
        max_iter=bayes_n[model_name][0],
        early_stop_iter=bayes_n[model_name][1],
        record_t=BO_record
    )
    BO.fit(x, y)
    BO.results_saving(sub_name, fg_name, model_name)
    best_params = BO.best_params_
    # 数据划分, 随机划分10次
    # 数据记录
    results_record = list()
    results_name = list()
    record_info = list()

    model = BO.best_estimator_

    model_kf = copy.deepcopy(model)

    best_model = str(best_params)
    foldX = KFold(n_splits=10, shuffle=True, random_state=100)
    epoch = 0

    for train_index, test_index in foldX.split(x):
        # train_results
        x_train, y_train = x.iloc[train_index, :], y[train_index]
        x_test, y_test = x.iloc[test_index, :], y[test_index]

        results_record += []
        results_name += []

        # over sample
        # s = OverSample()
        # u = RandomUnderSampler()
        # x_train, x_test = u.fit_resample(x_train, x_test)
        # x_train, x_test = s(x_train, x_test)

        model_kf.fit(x_train, y_train)
        pred_y = model_kf.predict(x_test)
        prob_y = model_kf.predict_proba(x_test)[:, 1]

        print('Train_results epoch: ', epoch)
        model_results, columns = metric(y_test, pred_y, prob_y)
        results_record += [model_results]
        record_info += [['train', '{}'.format(sub_name), '{}'.format(fg_name), '{}'.format(model_name), epoch]]
        epoch += 1

    print("\nModel_test")
    model.fit(x, y)
    # model_test
    t_pred_y = model.predict(t_x)
    t_prob_y = model.predict_proba(t_x)[:, 1]
    model_results, columns = metric(t_y, t_pred_y, t_prob_y)
    results_record += [model_results]
    record_info += [['test', '{}'.format(sub_name), '{}'.format(fg_name), '{}'.format(model_name), epoch]]

    with open('results_save/model/{}_{}_{}.pickle'.format(sub_name, fg_name, model_name), 'wb') as F:
        pickle.dump(model, F)

    return results_record, record_info, ['class', 'subtype_name', 'finger_name', 'model_name',
                                         'epoch'] + columns, best_model
    # 分布分析
    # distribute_analysis(x, y, t_x, t_y)


class OverSample(object):
    def __init__(self, method: str = 'B-SMOTE',
                 random_state: int = 10):
        self.method = method
        self.random_state = random_state

    def __call__(self, *args, **kwargs):
        if self.method == 'SMOTE':
            return self.SMOTE(*args)
        elif self.method == 'B-SMOTE':
            return self.B_SMOTE(*args)
        elif self.method == 'ADASYN':
            return self.ADASYN(*args)
        else:
            print('Method is error!')
            _ = ValueError

    def SMOTE(self, *args):
        smt = SMOTE(random_state=self.random_state)
        return smt.fit_resample(*args)

    def B_SMOTE(self, *args):
        bsmt = BorderlineSMOTE(random_state=self.random_state)
        return bsmt.fit_resample(*args)

    def ADASYN(self, *args):
        ada = ADASYN(random_state=self.random_state)
        return ada.fit_resample(*args)


def metric(y, y_pred, y_proba):
    auc = roc_auc_score(y, y_proba)
    acc = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    pre = precision_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spe = tn / (tn + fp)

    results = np.around(np.array([auc, acc, recall, pre, spe, mcc]), 4)
    columns = ['AUC', 'B_ACC', 'RECALL', 'PRE', 'SPE', 'MCC']
    print('=' * 40)
    print(["{}: {}".format(a, b) for a, b in zip(columns, list(results))])
    return results, columns


def main(
        combine: str = 'complete',
        if_des: bool = True
):
    model_name = list(_model.keys())
    for sub in subtype:
        sub_record = None
        model_record = None
        mdx = 0
        for mn in model_name:
            for fdx, fi in enumerate(finger_list):
                print('====== Processing: Model:{}   subtype:{}   fingerprint:{} ====== '.format(mn, sub, fi))
                train_path = 'datasets/Dataset_In_{}/{}_train_{}.csv'.format(fi, sub, fi)
                tr_desc_path = 'datasets/Dataset_In_MOLD2/{}_train_MOLD2.csv'.format(sub)
                test_path = 'datasets/Dataset_In_{}/{}_test_{}.csv'.format(fi, sub, fi)
                te_desc_path = 'datasets/Dataset_In_MOLD2/{}_test_MOLD2.csv'.format(sub)
                # _des_name = "{}_{}".format(fi, "MOLD2")
                d = DataProcessing(p_name=sub, d_name=fi)
                train_x, train_y = d(train_path)
                test_x, test_y = d.test_set(test_path)
                model_results, info, columns, best_model_info = train(train_x, train_y, test_x, test_y, mn,
                                                                      sub, fi)
                x1 = pd.DataFrame(info, columns=columns[:5])
                x2 = pd.DataFrame(np.array(model_results), columns=columns[5:])
                full_concat = pd.concat([x1, x2], axis=1)
                if mdx == 0:
                    model_record = pd.DataFrame([[sub, mn, fi, best_model_info]], columns=['Subtype_name',
                                                                                           'Model_name',
                                                                                           'Finger_name',
                                                                                           'Params'])
                    mdx = 1
                    sub_record = full_concat
                else:
                    model_record = pd.concat([model_record, pd.DataFrame([[sub, mn, fi, best_model_info]],
                                                                         columns=['Subtype_name',
                                                                                  'Model_name',
                                                                                  'Finger_name',
                                                                                  'Params'])], axis=0)
                    sub_record = pd.concat([sub_record, full_concat], axis=0)
        sub_record.to_csv('results_save/{}_complete_results.csv'.format(sub), index=False)
        model_record.to_csv('results_save/{}_complete_model_info.csv'.format(sub), index=False)


if __name__ == '__main__':
    main()
