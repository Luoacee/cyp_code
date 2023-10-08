import os.path
import pickle
import sklearn
import numpy as np
import pandas
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
from boruta import BorutaPy
from sklearn.feature_selection import mutual_info_regression
from typing import Tuple
import pandas as pd



def variance_screening(data: pd.DataFrame):
    vt = VarianceThreshold(threshold=0)
    vt.fit(data)
    return vt.get_feature_names_out()


def validation_strip(data: pd.DataFrame) -> Tuple[pd.DataFrame, np.numarray]:
    return data.iloc[:, 2:], np.array(data.iloc[:, 1].values)


def data_strip(data: pd.DataFrame) -> Tuple[pd.DataFrame, np.numarray]:
    return data.iloc[:, 3:], np.array(data.iloc[:, 2].values)


def feature_normalization(data):
    mms = MinMaxScaler()
    return pd.DataFrame(mms.fit_transform(data), columns=data.columns)


def normalization(function):
    # only for training set
    def wrapper(data, des_name, sub_name):
        if not os.path.exists("training_feature_save/normalization_{}_{}.tmp".format(sub_name, des_name)):
            model = function(data)
            with open("training_feature_save/normalization_{}_{}.tmp".format(sub_name, des_name), "wb") as file:
                pickle.dump(model, file)
            columns = data.columns.values.tolist()
            return_data = pd.DataFrame(model.transform(data), columns=columns)
        else:
            with open("training_feature_save/normalization_{}_{}.tmp".format(sub_name, des_name), "rb") as file:
                model = pickle.load(file)
            columns = data.columns.values.tolist()
            return_data = pd.DataFrame(model.transform(data), columns=columns)
        return return_data
    return wrapper


@normalization
def normalization(data: pd.DataFrame):
    mms = MinMaxScaler()
    mms.fit(data)
    return mms


def feature_split(data: pd.DataFrame):
    all_columns = data.columns.tolist()
    c_c = list()
    for c_name in all_columns:
        # print(data.loc[:, c_name].values.tolist())
        record_int = 0
        for i in data.loc[:, c_name].values.tolist():
            if abs(i - int(i)) > 9e-8:
                c_c += [c_name]
                break
    c_data = data.loc[:, c_c]
    # c data
    d_d = list()
    for d_name in all_columns:
        if d_name in list(set(all_columns) - set(c_c)):
            d_d += [d_name]
    d_data = data.loc[:, d_d]
    if len(c_c) == 0:
        c_data = None
    elif len(d_d) == 0:
        d_data = None
    return d_data, c_data


def _data_prepare(*args: str):
    if args[-1] == "train" or args[-1] == "test":
        if len(args) == 2:
            loadA = pd.read_csv(args[0])
            X, Y = data_strip(loadA)
            return X, Y
        elif len(args) == 3:
            loadA, loadB = pd.read_csv(args[0]), pd.read_csv(args[1])
            xA, yA = data_strip(loadA)
            xB, yB = data_strip(loadB)
            assert len(yA) == len(yB), "Data loading error"
            return pd.concat([xA, xB], axis=1), yA
    elif args[-1] == "valid":
        if len(args) == 2:
            loadA = pd.read_csv(args[0])
            X, Y = validation_strip(loadA)
            return X, Y
        elif len(args) == 3:
            loadA, loadB = pd.read_csv(args[0]), pd.read_csv(args[1])
            xA, yA = validation_strip(loadA)
            xB, yB = validation_strip(loadB)
            assert xA == xB, "Data loading error"
            return pd.concat([xA, xB], axis=1), yA


def train_load(*args: str):
    trainX, trainY = _data_prepare(*args, "train")
    FeatureX = variance_screening(trainX)
    trainX = trainX.loc[:, FeatureX]
    return trainX, trainY


def test_valid_load(*args: str):
    if args[-1] == "test":
        X, Y = _data_prepare(*args)
        return X, Y
    elif args[-1] == "valid":
        X, Y = _data_prepare(*args)
        return X, Y
    else:
        _ = ValueError


def genetic_method(model, tr_x, tr_y, te_x, te_y):
    selector = GeneticSelectionCV(model,
                                  cv=10,
                                  scoring="accuracy",
                                  # max_features=15,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1
                                  )
    selector.fit(tr_x.values, tr_y)
    re = 0
    print(selector.support_)
    for i in selector.support_:
        if bool(i) is True:
            re += 1
    print(re)

def _mul_info(x, y, k: float = 0.8):
    mu = mutual_info_regression(x, y, discrete_features=False, random_state=10)
    mul_sort = sorted(range(len(mu)), key=lambda l1: mu[l1], reverse=True)
    mul_sort = mul_sort[:int(len(mul_sort)*k)]
    mul_data = x.iloc[:, mul_sort]
    return mul_data, mul_data.columns.tolist()


def boruta_py(model, tr_x, tr_y, te_x, te_y):
    brt = BorutaPy(estimator=model,
                   max_iter=100)
    brt.fit(np.array(tr_x), tr_y)
    print(len(tr_x.columns))
    valid_columns = tr_x.columns[brt.support_].tolist()
    print(len(valid_columns))


def normalization_other(model, data):
    columns_ = data.columns
    data = model.transform(data)
    return pd.DataFrame(data, columns=columns_)

def mult_full(dataX: pd.DataFrame, dataY: np.numarray, des_name, sub_name
              ) -> None:
    d_data, c_data = feature_split(dataX)
    try:
        normalC = feature_normalization(c_data)
        normalD = feature_normalization(d_data)
        multD = _mul_info(normalD, dataY)
        multC = _mul_info(normalC, dataY)
        fDataX = pd.concat([multC[0], multD[0]], axis=1)
    except (ValueError, AttributeError):
        try:
            normal = feature_normalization(d_data)
        except ValueError:
            normal = feature_normalization(c_data)
        fDataX = _mul_info(normal, dataY)[0]

    with open("training_feature_save/feature_name_{}_{}.tmp".format(sub_name, des_name), 'wb') as f:
        pickle.dump(fDataX.columns.values.tolist(), f)


def test_processing(dataX: pd.DataFrame, dataY: np.numarray, endpoint_name, des_name):
    with open("training_feature_save/feature_name_{}_{}.tmp".format(endpoint_name, des_name), 'rb') as f:
        valid_index = pickle.load(f)
    with open("training_feature_save/normalization_{}_{}.tmp".format(endpoint_name, des_name),
              "rb") as file:
        normal_model = pickle.load(file)
    return normalization_other(normal_model, dataX.loc[:, valid_index]), dataY


class DataProcessing(object):
    def __init__(self, p_name, d_name):
        self.point_name = p_name
        self.des_name = d_name
        self.feature_save = "training_feature_save/feature_name_{}_{}.tmp".format(self.point_name, self.des_name)

    def __call__(self, *args):
        trX, trY = train_load(*args)
        if not os.path.exists("training_feature_save/feature_name_{}_{}.tmp".format(self.point_name, self.des_name)):
            mult_full(trX, trY, self.des_name, self.point_name)

        with open("training_feature_save/feature_name_{}_{}.tmp".format(self.point_name, self.des_name), 'rb') as fn:
            feature_names = pickle.load(fn)
        trX = trX.loc[:, feature_names]
        trX = normalization(trX, self.des_name, self.point_name)
        return trX, trY

    def test_set(self, *args):
        teX, teY = test_valid_load(*args, "test")
        teX, teY = test_processing(teX, teY, self.point_name, self.des_name)
        return teX, teY

    def valid_set(self, *args):
        vaX, vaY = test_valid_load(*args, "valid")
        vaX, vaY = test_processing(vaX, vaY, self.point_name, self.des_name)
        return vaX, vaY


if __name__ == "__main__":
    pass


