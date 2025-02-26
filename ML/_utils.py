import copy
import os
import pandas as pd
# sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import random
random.seed(10)
# 對訓練集和測試集採用去重的操作


# =====================================================================================================================
#                                                   Data processing
# =====================================================================================================================
def first_check(dataset: pd.DataFrame = None,
                remove_na: list = 'mean',
                threshold: float = 0.05,
                valid_range_start: int = 5,
                remove_invalid_list: list = None):

    value_r = None
    remove_invalid_list = ['Ipc', 'index'] + ['index.{}'.format(i) for i in range(1, 11)]

    def _drop_column():
        _valid_column = list()
        for i in remove_invalid_list:
            if i in dataset.columns.tolist():
                _valid_column += [i]
        return _valid_column

    # 進行基本的去重去空值操作
    print('Before processing: line: {} column: {}'.format(len(dataset), len(dataset.columns)))

    # 去除空值及對應列
    dataset.dropna(axis=1, inplace=True, how='all')
    valid_column = _drop_column()
    dataset.drop(columns=valid_column, inplace=True)

    # 空值處理, 當空值數量高於5%, 直接捨去，否則使用均值取代/0取代
    valid_column = [c_n for c_n in dataset.columns.tolist()
                    if dataset[c_n].isna().sum() < len(dataset)*threshold]
    dataset = dataset.loc[:, valid_column]

    # 空值填充，使用均值或者0

    if remove_na == 'mean':
        value_r = dict([(c_name, c_value) for c_name, c_value
                        in zip(dataset.iloc[:, valid_range_start:].columns.tolist(),
                               dataset.iloc[:, valid_range_start:].mean().tolist())])
    elif remove_na == 'zero':
        value_r = {c_n: 0 for c_n in dataset.iloc[:, valid_range_start:].columns.tolist()}
    else:
        print('Remove choice is error: mean/zero')
        _ = ValueError
    dataset.fillna(value=value_r, inplace=True)
    dataset.reset_index(inplace=True)

    # 去除列
    valid_column = _drop_column()
    dataset.drop(columns=valid_column, inplace=True)

    print('After processing: line: {} column: {}'.format(len(dataset), len(dataset.columns)))
    return dataset


# 針對老csv文件列進行先行處理

def temporary_api(path: str = None, dataset: pd.DataFrame = None):
    remove_invalid_list = ['index'] + ['index.{}'.format(i) for i in range(1, 11)]
    _file_read = dataset

    def _drop_column():
        _valid_column = list()
        for i in remove_invalid_list:
            if i in _file_read.columns.tolist():
                _valid_column += [i]
        return _valid_column

    valid_column = _drop_column()
    _file_read.drop(columns=valid_column, inplace=True)
    processed = None

    if 'train_data' in path:
        info_c = ['PubChem_cid', 'Smiles', 'Y']
        processed = extract_columns(info_c, _file_read)
    elif 'test_data' in path:
        info_c = ['SMILES', 'Y']
        processed = extract_columns(info_c, _file_read)
    else:
        print('Path error')
        _ = ValueError
    # 可以選擇將原有的文件替換，目前不可
    # processed.to_csv(path)
    return processed


def extract_columns(c_n, data):
    data_column = data.loc[:, c_n[-1]:]
    data_column.drop(columns=c_n[-1], inplace=True)
    return pd.concat((data.loc[:, c_n], data_column), axis=1)


def contemporary_concat(*args):
    # 根據輸入數據的數量，判斷是否要進行拼接，輸入一個，不拼接，輸入兩個進行拼接，拼接後共同進入檢查函數，
    # 最終返回一個去重的值，拼接時將其中一個的重複值
    # 數據同意，第一列為cis, 第二列為smiles, 第三列為y, 其他列都為數據列
    if len(args) == 1:
        p = process_pip(args[0], pd.read_csv(args[0]))
        return p
    elif len(args) == 2:
        dc = pd.concat((pd.read_csv(args[0]), pd.read_csv(args[1])), axis=0)
        dc.reset_index(inplace=True, drop=True)
        p = process_pip(args[0], dc)
        return p


def simply_concat(a, b):
    # 仅用于测试集, 描述符拼接
    file_a = pd.read_csv(a)
    file_b = pd.read_csv(b)
    return pd.concat([file_a, file_b], axis=1)


def simply_processing(data, column_select):
    # 仅用于测试集的简单处理, 按照column_name进行提取，对空值采用填充均值的方式处理
    t_data = data.loc[:, column_select]
    temp_data = copy.deepcopy(data)
    test_c = data.columns.tolist()
    for i in ['index'] + ['index.{}'.format(j) for j in list(range(1, 11))]:
        if i in test_c:
            temp_data.drop(columns=[i], inplace=True)

    value_t = dict([(c_name, c_value) for c_name, c_value
                    in zip(t_data.columns.tolist(),
                           t_data.mean().tolist())])

    t_data.fillna(value=value_t, inplace=True)
    return pd.concat([temp_data.iloc[:, :2], t_data], axis=1)


def heterogeneous_concat(*args):
    # 将不同源的数据进行拼接，如指纹+描述符拼接
    f1 = process_pip(args[0], pd.read_csv(args[0]))
    f2 = process_pip(args[1], pd.read_csv(args[1]))

    # return f1, f2
    # 不参与拼接
    if 'train_data' in args[0]:
        return pd.concat((f1, f2.iloc[:, 3:]), axis=1)
    elif 'test_data' in args[0]:
        return pd.concat((f1, f2.iloc[:, 2:]), axis=1)


def process_pip(file_path, dataset):
    # 先將數據列表進行標準化, 標准列提取
    extract_csv = temporary_api(file_path, dataset)
    # 對數據採用去重等操作
    return first_check(extract_csv)


def dc_feature_splitting(data):
    # d data
    all_columns = data.columns.tolist()
    c_c = list()
    for c_name in all_columns:
        # print(data.loc[:, c_name].values.tolist())
        record_int = 0
        for i in data.loc[:, c_name].values.tolist():
            if i - int(i) > 9e-8:
                c_c += [c_name]
                break

    c_data = data.loc[:, c_c]
    # c data
    d_d = list()
    for d_name in all_columns:
        if d_name in list(set(all_columns)-set(c_c)):
            d_d += [d_name]
    d_data = data.loc[:, d_d]
    # 返回离散数据，连续数据
    return d_data, c_data


file_read_p = 'datasets/Dataset_MACCS/train_data/cyp2c9_A_MACCS.csv'
file_read_p2 = 'datasets/Dataset_D_RDKIT/train_data/cyp2c9_A_rdkit.csv'
# # 先將數據列表進行標準化, 標准列提取
# extract_csv = temporary_api(file_read_p)
# # 對數據採用去重等操作
# d = first_check(extract_csv)

# 对相同数据(A, B)进行拼接
# train_data = contemporary_concat(file_read_p)

# 将不同类的数据进行拼接，
# x = heterogeneous_concat(file_read_p2, file_read_p)
# x.to_csv('test_feature_save.csv', index=False)
# 以上功能,輸入一個地址 或兩個地址，得到標準後的pandas數據，也就是去重，去除空值/缺失值的數據
# ======================================================================================================================


# =====================================================================================================================
#                                          Normalization (MinMax, Standard)
# =====================================================================================================================

class Scalar(object):
    def __init__(self, options: str = 'MinMax'):
        """
        Minmax
        Standard
        """
        self.option = options

    def __call__(self, data):
        if self.option == 'MinMax':
            return self._min_max_scalar(data)
        elif self.option == 'Standard':
            return self._standard_scalar(data)
        else:
            print("Values error")
            _ = ValueError

    @staticmethod
    def _min_max_scalar(data):
        sc = MinMaxScaler()
        return sc.fit_transform(data)

    @staticmethod
    def _standard_scalar(data):
        sc = StandardScaler()
        return sc.fit_transform(data)


# =====================================================================================================================
#                                                    Feature select
# =====================================================================================================================

# ================================================================================
#                    Filter Method (Var, Chi2, Mutual_info_regression)
# ================================================================================


def var_threshold(td: pd.DataFrame = None):
    # 此时只进行0值筛选
    vt = VarianceThreshold(threshold=0)
    vt.fit(td)
    return vt.transform(td), vt.get_feature_names_out()


def _chi2(x, y, k: float = 0.8):
    ch = SelectKBest(chi2, k=int(len(x.columns)*k))
    ch.fit_transform(x, y)
    ch_data = x.loc[:, ch.get_feature_names_out()]
    return ch_data, ch.get_feature_names_out()


def _mul_info(x, y, k: float = 0.8):
    mu = mutual_info_regression(x, y, discrete_features=False, random_state=10)
    mul_sort = sorted(range(len(mu)), key=lambda l1: mu[l1], reverse=True)
    mul_sort = mul_sort[:int(len(mul_sort)*k)]
    mul_data = x.iloc[:, mul_sort]
    return mul_data, mul_data.columns.tolist()

# =====================================================================================================================
#                                                Data processing pip
# =====================================================================================================================

#           2个 1:1        3个 2:1              4个 2:2                 7个 4:3
# 输入包括 单数据集 数据量拼接数据集 不同描述符拼接数据集  数据量+描述符拼接数据集  都包括训练集和测试集
# 当仅有指纹时，不进行
# 自动判断

# 输入数据，选择是否进行描述符种类划分。当不划分时，默认为使用离散特征，只采用VT以及卡方特征筛选。
# 当划分描述符时，采用两种特征，卡方以及信息增益方式进行特征筛选。


# def training_pretreatment(file_read_p, file_read_p2):
#     # --------------------- Data concat -----------------------
#     concat_data = heterogeneous_concat(file_read_p2, file_read_p)
#
#     # --------------------- Discrete and continuous data split -----------------------
#     Y_info = concat_data.loc[:, 'Y']
#     discrete, continuous = dc_feature_splitting(concat_data.iloc[:, 3:])
#
#     # --------------------- Remove redundant column -----------------------
#     dis_f = var_threshold(discrete)
#     con_f = var_threshold(continuous)
#     dis_f_pd = pd.DataFrame(np.array(dis_f[0]), columns=dis_f[1])
#     con_f_pd = pd.DataFrame(np.array(con_f[0]), columns=con_f[1])
#
#     # --------------------- Normalization of continuous data -----------------------
#     S = Scalar('Standard')
#     scaler_file = S(con_f_pd)
#     con_f_stand = pd.DataFrame(np.array(scaler_file), columns=con_f_pd.columns.tolist())
#     dis_f_stand = dis_f_pd
#
#     # --------------------- Feature selection -----------------------
#     dis_features, dis_name = _chi2(dis_f_stand, Y_info)
#     con_features, con_name = _mul_info(con_f_stand, Y_info)
#
#     # --------------------- Final concat -----------------------
#     final_out = pd.concat([concat_data.iloc[:, :3], con_features, dis_features], axis=1)
#     return final_out

def _pip(data, split_des=True):
    Y_info = data.loc[:, 'Y']

    if split_des:
        discrete, continuous = dc_feature_splitting(data.iloc[:, 1:])
        dis_f = var_threshold(discrete)
        con_f = var_threshold(continuous)
        dis_f_pd = pd.DataFrame(np.array(dis_f[0]), columns=dis_f[1])
        con_f_pd = pd.DataFrame(np.array(con_f[0]), columns=con_f[1])

        S = Scalar('Standard')
        scaler_file = S(con_f_pd)
        con_f_stand = pd.DataFrame(np.array(scaler_file), columns=con_f_pd.columns.tolist())

        dis_f_stand = dis_f_pd
        print('Confusion processing!')
        dis_features, dis_name = _chi2(dis_f_stand, Y_info)
        con_features, con_name = _mul_info(con_f_stand, Y_info)

        final_out = pd.concat([con_features, dis_features], axis=1)
        return final_out

    else:
        r_y = data.iloc[:, 1:]
        var_p = var_threshold(r_y)
        r_pd = pd.DataFrame(np.array(var_p[0]), columns=var_p[1])
        mk = 1
        for i in r_pd.iloc[:, 1].values.tolist():
            if abs(int(i) - i) > 9e-8:
                mk = 0
                break
        if mk:
            print('Chi2 processing!')
            r_features, r_name = _chi2(r_pd, Y_info)
        else:
            print('Mul_info_processing!')
            r_features, r_name = _mul_info(r_pd, Y_info)
        return r_features


def option_process(*args):
    # feature processing
    # 单一数据
    if len(args) == 2:
        # - 训练集
        single_d_t = contemporary_concat(args[0])

        try:
            _single_d_t = _pip(single_d_t.iloc[:, 2:], split_des=True)
        except ValueError:
            _single_d_t = _pip(single_d_t.iloc[:, 2:], split_des=False)
        # _single_d_t = single_d_t.iloc[:, 3:]
        # - 测试集
        single_d_e = simply_processing(pd.read_csv(args[1]), _single_d_t.columns.tolist())
        return pd.concat([single_d_t.iloc[:, :3], _single_d_t], axis=1), single_d_e

    # 数据量增加的数据
    elif len(args) == 3:
        multi_d_t = contemporary_concat(args[0], args[1])

        try:
            _multi_d_t = _pip(multi_d_t.iloc[:, 2:], split_des=True)
        except ValueError:
            _multi_d_t = _pip(multi_d_t.iloc[:, 2:], split_des=False)

        multi_d_e = simply_processing(pd.read_csv(args[2]), _multi_d_t.columns.tolist())
        return pd.concat([multi_d_t.iloc[:, :3], _multi_d_t], axis=1), multi_d_e

    # 数据量拼接的数据
    elif len(args) == 4:
        multi_c_t = heterogeneous_concat(args[0], args[1])
        _multi_c_t = _pip(multi_c_t.iloc[:, 2:], split_des=True)

        multi_c_e = simply_concat(args[2], args[3])
        multi_c_e = simply_processing(multi_c_e, _multi_c_t.columns.tolist())
        return pd.concat([multi_c_t.iloc[:, :3], _multi_c_t], axis=1), multi_c_e

    # 全拼接数据
    elif len(args) == 6:
        # 0, 1, 2, 3    0: descA, 1: descB, 2: fingerA, 3:fingerB
        multi_1 = contemporary_concat(args[0], args[1])
        multi_2 = contemporary_concat(args[2], args[3])

        multi_f_t = pd.concat([multi_1, multi_2.iloc[:, 3:]], axis=1)
        _multi_f_t = _pip(multi_f_t.iloc[:, 2:], split_des=True)

        multi_f_e = simply_concat(args[4], args[5])
        multi_f_e = simply_processing(multi_f_e, _multi_f_t.columns.tolist())
        return pd.concat([multi_f_t.iloc[:, :3], _multi_f_t], axis=1), multi_f_e
    else:
        print('Path error')
        _ = ValueError
