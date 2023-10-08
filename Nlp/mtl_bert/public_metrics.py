from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import numpy as np


def pos_neg_threshold(function):
    def wrapper(y, y_proba, threshold=0.5):
        y_pred = np.where(np.array(y_proba) >= threshold, 1, 0)
        results, columns = function(y, y_pred, y_proba)
        return results, columns
    return wrapper


@pos_neg_threshold
def metric(y, y_pred, y_proba):
    auc = roc_auc_score(y, y_proba)
    acc = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    pre = precision_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spe = tn / (tn + fp)

    results = np.around(np.array([auc, acc, recall, spe, pre, mcc]), 4)
    columns = ['AUC', 'B_ACC', 'RECALL', 'SPE', 'PRE', 'MCC']
    print('=' * 40)
    print(["{}: {}".format(a, b) for a, b in zip(columns, list(results))])
    return results, columns

