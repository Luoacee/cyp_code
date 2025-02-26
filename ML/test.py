# 参数加载
import os
import pickle
import pandas as pd
from model import metric
import numpy as np

def valid(cyp_name, finger_name, model_name, train_x, train_y, valid_x, valid_y):
    with open("model_save/%s_%s_%s.pickle" % (cyp_name, finger_name, model_name), "rb") as F:
        model = pickle.load(F)
    model.fit(train_x, train_y)
    pred_y = model.predict(valid_x)
    prob_y = model.predict_proba(valid_x)[:, 1]
    model_results, columns = metric(valid_y.values.tolist(), pred_y, prob_y)
    record_info = [['exvalid', '{}'.format(cyp_name), '{}'.format(finger_name), '{}'.format(model_name)]]
    return model_results, record_info, ['class', 'subtype_name', 'finger_name', 'model_name',
                                         ] + columns

def main():
    cyp_name = ["cyp2c9", "cyp2d6", "cyp3a4"]
    model_name = ["Xgboost", "Xgboost", "Catboost"]
    finger_name = ["MORGAN-F_RDKIT", "MORGAN_MOLD2", "MORGAN-F_RDKIT"]
    mdx = 0
    sub_record = None
    for c, m, f in zip(cyp_name, model_name, finger_name):
        f1, f2 = f.split("_")
        v1_load = pd.read_csv("datasets/Dataset_In_%s/test_data/%s_%s.csv" % (f1, c.upper(), f1))
        v2_load = pd.read_csv("datasets/Dataset_In_%s/test_data/%s_%s.csv" % (f2, c.upper(), f2))
        # test1 = pd.read_csv("datasets/Dataset_In_MORGAN-F/cyp2c9_test_MORGAN-F.csv")
        # test2 = pd.read_csv("datasets/Dataset_In_RDKIT/cyp2c9_test_RDKIT.csv")
        train_1 = pd.read_csv("datasets/Dataset_In_%s/%s_train_%s.csv" % (f1, c, f1))
        train_2 = pd.read_csv("datasets/Dataset_In_%s/%s_train_%s.csv" % (f2, c, f2))
        with open("model_save/feature_name_%s_%s.tmp" % (c, f), "rb") as F:
            choice_columns = pickle.load(F)
        vf = pd.concat([v1_load, v2_load], axis=1)
        # tf = pd.concat([test1, test2], axis=1)
        trf = pd.concat([train_1, train_2], axis=1)
        valid_x = vf.loc[:, choice_columns]
        # test_x = tf.loc[:, choice_columns]
        train_x = trf.loc[:, choice_columns]
        valid_y = vf["Y"].iloc[:, 0]
        # test_y = tf["Y"].iloc[:, 0]
        train_y = trf["Y"].iloc[:, 0]
        model_results, info, columns = valid(c, f, m, train_x, train_y, valid_x, valid_y)
        x1 = pd.DataFrame(info, columns=columns[:4])
        x2 = pd.DataFrame([model_results], columns=columns[4:])
        full_concat = pd.concat([x1, x2], axis=1)
        print(full_concat)
        if mdx == 0:
            mdx = 1
            sub_record = full_concat
        else:
            sub_record = pd.concat([sub_record, full_concat], axis=0)
        sub_record.to_csv('exvalid_complete_results.csv'.format(c), index=False)

main()

