import functools
import numpy as np
from functools import wraps
from functools import partial
import pandas as pd
import tqdm
import pickle
from molecular_simply import Process
from rdkit import Chem
from typing import Union
from typing import Tuple
import re
import random
import os

# ===================================
#            some params
# ===================================
ATOM = ['C', 'H', 'O', 'N', 'P', 'S', 'B', 'F', "Cl", "Br", "I", "Si", "Mg", "Ca", "Fe", "As", "Al"]
SMILES_CHAR = ['c', 'o', 'n', 's', 'p', '=', '(', ')', '[', ']', '#', '@', '%', '/', '\\', '+', '-', '0',
               '1', '2', '3', '4', '5', '6', '7', '8', '9']
SEMANTIC = ["<p0>", "<PAD>", "<MASK>", "<SOS>", "<EOS>", "<CLS>", "<UNK>"]
PAT = r"Br|Cl|Si|Mg|Ca|Fe|As|Al|[1234567890]|[CHONPSBFIcons]|[#%@=\\\\/\+\-\(\)\[\]]"
SUPPLEMENTARY = ["<p{}>".format(i) for i in range(1, 20)]
FULL_LIST = SEMANTIC + ATOM + SMILES_CHAR + SUPPLEMENTARY
c2i = {c: idx for idx, c in enumerate(FULL_LIST)}
i2c = {idx: c for c, idx in c2i.items()}


def molecular_initiation(smiles: Union[np.ndarray, list, pd.DataFrame] = None, atom_list: list = None,
                         length_limit: int = 500, smiles_shuffle: bool = False) -> Union[list, pd.DataFrame]:
    p = Process(atom_list=atom_list, re_hydro=True, defn_list='[Na,K,Mg,I,Cl,Br,Ca,Fe,Al,As]', atom_number_limit=120,
                shuffle_atoms=smiles_shuffle)

    def _limit(_smiles: list = None) -> list:
        _limit_list = list()
        for i in _smiles:
            if i is not None and len(i) <= length_limit:
                _limit_list += [i]
            else:
                _limit_list += [None]
        return _limit_list

    if isinstance(smiles, (list, np.ndarray)):
        print("Input SMILEs:", len(smiles))
        returnSMILES = p(smiles)
        returnSMILES = _limit(returnSMILES)
        while True:
            if None in returnSMILES:
                returnSMILES.remove(None)
            else:
                break
        return returnSMILES

    elif isinstance(smiles, pd.DataFrame):
        print("Input SMILEs:", len(smiles))
        smi_list = smiles["Smiles"].values.tolist()
        returnSMILES = p(smi_list)
        returnSMILES = _limit(returnSMILES)
        smiles.Smiles = returnSMILES
        smiles.dropna(subset=["Smiles"], inplace=True)
        smiles.reset_index(drop=True, inplace=True)
        return smiles
    else:
        print("Type error")
        returnSMILES = ValueError


def shuffle_smiles(smiles_: str = None) -> Union[str, None]:
    mol = Chem.MolFromSmiles(smiles_)
    if not mol:
        return None
    else:
        atom_n = mol.GetAtoms()
        atom_index = list(range(len(atom_n)))
        random.shuffle(atom_index)
        shuffle_mol = Chem.RenumberAtoms(mol, atom_index)
        new_smiles = Chem.MolToSmiles(shuffle_mol, canonical=False)
        return new_smiles


def file_io(*args, path, io_class: str = "write") -> Union[None, list, np.ndarray]:
    if io_class == "write":
        with open(path, "wb") as file:
            pickle.dump(args, file)
    else:
        with open(path, "rb") as file:
            load_data = pickle.load(file)
        return load_data


def unsupervised_processing(function):
    @wraps(function)
    def wrapper(data_: list = None,
                p_: functools.partial = None,
                mask: bool = False,
                mask_proportion: float = 0.15,
                encoding_length: int = None,
                save_file_name: str = "data/processed_file.pickle",
                re_make: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list]:
        save_file_path = "{}".format(save_file_name)
        if (os.path.exists(save_file_path) is False) or (re_make is True):
            print("Input Smiles :", len(data_))
            print("Unsupervised processing ->")
            smiles_ = data_
            # smiles_ = p_(smiles=data_)
            # print("Valid Smiles :", len(smiles_))
            results = tqdm.tqdm(map(lambda la: function(la, mask, mask_proportion), smiles_), total=len(smiles_))
            if bool(encoding_length):
                x, y, w, lth = [], [], [], []

                def completion(seq, length) -> list:
                    seq += [c2i["<PAD>"] for _ in range((length - len(seq)))]
                    assert len(seq) == length, "Completion error:{}".format(seq)
                    return seq

                for r in results:
                    x += [completion(r[0], length=encoding_length)]
                    y += [completion(r[1], length=encoding_length)]
                    w += [completion(r[2], length=encoding_length)]
                    lth.append(r[3])
                x, y, w, lth = np.array(x, dtype=int), np.array(y, dtype=int), \
                    np.array(w, dtype=float), np.array(lth, dtype=int)
                file_io(x, y, w, lth, path=save_file_path)
                return x, y, w, lth
            file_io(results, path=save_file_path)
            return list(results)
        else:
            print("Load file ->")
            return file_io(None, path=save_file_path, io_class="read")

    return wrapper


def char2idx(function):
    @wraps(function)
    def wrapper(smiles_: str = None):
        charList = re.findall(PAT, smiles_)
        encoding_ = function(charList)
        # check encoding
        idx2char_ = [i2c[i] for i in encoding_]
        assert idx2char_ == charList, "Char2idx encoding error!"
        return encoding_
    return wrapper


@char2idx
def char2idx(char_list: list = None) -> list:
    idx = [c2i[i] for i in char_list]
    return idx


def supervised_processing(function):
    @wraps(function)
    def wrapper(data_: list = None,
                p_: functools.partial = None,
                encoding_length: int = None,
                save_file_name: str = "data/supervised_processed_file.pickle",
                re_make: bool = False) -> Union[Tuple[list, np.ndarray, np.ndarray, np.ndarray], list]:
        save_file_path = "{}".format(save_file_name)
        if (os.path.exists(save_file_path) is False) or (re_make is True):
            print("Input Smiles :", len(data_))
            print("Supervised processing ->")
            smiles_ = p_(smiles=data_)
            # return pandas data
            print("Valid Smiles :", len(smiles_))

            # pandas ->
            smiles_data = smiles_["Smiles"].values.tolist()
            y = smiles_["Y"].values.tolist()
            results = tqdm.tqdm(map(lambda la: function(la), smiles_data), total=len(smiles_data))
            if bool(encoding_length):
                x, lth = [], []
                def completion(seq, length) -> list:
                    seq += [c2i["<PAD>"] for _ in range((length - len(seq)))]
                    assert len(seq) == length, "Completion error:{}".format(seq)
                    return seq

                for r in results:
                    x += [completion(r[0], length=encoding_length)]
                    lth.append(r[1])
                x, y, lth = np.array(x, dtype=int), np.array(y, dtype=int), np.array(lth, dtype=int)
                file_io(smiles_data, x, y, lth, path=save_file_path)
                return smiles_data, x, y, lth
            file_io(results, path=save_file_path)
            return list(results)
        else:
            print("Load file ->")
            return file_io(None, path=save_file_path, io_class="read")
    return wrapper


@supervised_processing
def supervised_processing(Smiles_: list = None):
    return_c = char2idx(Smiles_)
    x = return_c + [c2i["<SOS>"]]
    lth = len(return_c)
    return x, lth
# File read   => 无监督学习处理， 有监督处理


@unsupervised_processing
def unsupervised_processing(smiles_: str = None,
                            mask: bool = False,
                            mask_proportion: float = 0.2) -> Tuple[list, list, list, int]:
    # ------> length check
    return_c = char2idx(smiles_)
    y = [c2i["<SOS>"]] + return_c
    if mask:
        encoding_index = list(range(len(return_c)))
        random.shuffle(encoding_index)
        lth = len(return_c)
        w = [0.] * len(return_c)
        random_block = [0, 0, 0, 1, 0, 0, 1, 1]
        for i in encoding_index[:int(len(return_c)*mask_proportion)]:
            k = random.choice(random_block)
            if k == 0:
                return_c[i] = c2i["<UNK>"]
            else:
                return_c[i] = c2i[random.choice(ATOM)]
            w[i] = 1
        x = [c2i["<SOS>"]] + return_c
    else:
        lth = len(return_c)
        x = [c2i["<SOS>"]] + return_c
        w = [1.] * len(x)

    return x, y, w, lth


def gen(dir_):
    file_read = pd.read_csv("split/%s/Dataset_In_MACCS/cyp2c9_train_MACCS.csv" % dir_)
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    supervised_processing(file_read.iloc[:, 0:3], p, encoding_length=150,
                                                 save_file_name="split/%s/Dataset_In_MACCS/2c9_train.pkl" % dir_)

    file_read_ = pd.read_csv("split/%s/Dataset_In_MACCS/cyp2c9_test_MACCS.csv" % dir_)
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    supervised_processing(file_read_.iloc[:, 0:3], p, encoding_length=150,
                                                 save_file_name="split/%s/Dataset_In_MACCS/2c9_test.pkl" % dir_)

    file_read_ = pd.read_csv("split/%s/Dataset_In_MACCS/cyp2c9_valid_MACCS.csv" % dir_)
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    supervised_processing(file_read_.iloc[:, 0:3], p, encoding_length=150,
                                                 save_file_name="split/%s/Dataset_In_MACCS/2c9_valid.pkl" % dir_)


if __name__ == "__main__":
    # 输入SMILES
    # smiles = torch.load("500k_smiles.pth")[-1][:11000]

    # read_file = pd.read_csv("Valid_smiles_32.csv")
    # read_file.dropna(subset=["Smiles"], inplace=True)
    # smiles = read_file.Smiles.values.tolist()
    # valid_smiles = []
    # for i in smiles:
    #     if len(i) <= 145:
    #         valid_smiles += [i]

    # -> 无监督学习编码 -> 直接编码即可 -> 包含一个起始字符 ->  最终填充到一定的长度（填充字符）
    # -> 针对一个SMILES, 抽取20%的数据，使用mask掩码操作，当随机数<0.8, mask掉，当随机数为0.8-0.9之间，随机替换有效的数值，0.9-1不变。
    # -> 随机替换的数据，抽取到的数值，权重为1，抽取到的数据，权重为0,
    # p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    # rx, ry, rw, rl = unsupervised_processing(valid_smiles, p, mask=True, encoding_length=150,
    #                                          save_file_name="data/pad_test.pkl")
    # print(rx.shape, ry.shape, rw.shape)
    # print(rx[10])
    # print(ry[10])
    # print(rw[10])
    # print(rl[10])

    # Fine-tuning process
    file_read = pd.read_csv("cyp2c9_train_MACCS.csv")
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    smiles, x, y, length = supervised_processing(file_read.iloc[:, 0:3], p, encoding_length=150,
                                                 save_file_name="data/2c9_train.pkl")

    file_read_= pd.read_csv("cyp2c9_test_MACCS.csv")
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    smiles__, x__, y__, length__ = supervised_processing(file_read_.iloc[:, 0:3], p, encoding_length=150,
                                                         save_file_name="data/2c9_test.pkl")

    file_read_ = pd.read_csv("cyp2c9_valid_MACCS.csv")
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    smiles_, x_, y_, length_ = supervised_processing(file_read_.iloc[:, 0:3], p, encoding_length=150,
                                                         save_file_name="data/2c9_valid.pkl")
    file_read_ = pd.read_csv("data/evalid/cyp3a4_MACCS.csv")
    p = partial(molecular_initiation, atom_list=ATOM, smiles_shuffle=True, length_limit=145)
    smiles_, x_, y_, length_ = supervised_processing(file_read_.iloc[:, 0:2], p, encoding_length=150,
                                                         save_file_name="data/3a4_evalid.pkl")




