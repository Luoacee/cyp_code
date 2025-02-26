import random
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit.Chem import Descriptors
from rdkit.Chem import SaltRemover
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import os
import pandas as pd
import collections
from molvs import normalize, metal
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

random.seed(100)


def _map(*args):
    return list(map(*args))


def remove_mixture(smi: str = None):
    spt_smi = smi.split('.')
    length = list()
    for s in spt_smi:
        length.append(len(s))
    if (len(spt_smi) >= 2) and (len(spt_smi[np.argmax(length)])) > 5:
        max_index = np.argmax(np.array(length))
        return spt_smi[max_index]
    else:
        return smi


def add_hydrogen(mol, opt=False):
    if opt:
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)


def weight_filter(mol, scope: tuple = (0, 500)):
    mol = add_hydrogen(mol, True)
    wt = Descriptors.MolWt(mol)
    if scope[0] <= wt <= scope[1]:
        return add_hydrogen(mol)
    else:
        return None


def remove_salt(mol, desalt):
    return desalt(mol)


def remove_inorganic(mol):
    atom_info = mol.GetAtoms()
    atom_sym = list()
    if len(atom_info) <= 2:
        return None
    for i in atom_info:
        atom_sym.append(i.GetSymbol())
    atom_dict = collections.Counter(atom_sym)
    try:
        atom_dict['C'] != 0
    except KeyError:
        return None
    if atom_dict['C'] <= 1:
        return None
    return mol


def atom_screen(mol, atom_list: list = None, atom_num: int = None):
    atom_info = mol.GetAtoms()
    if len(atom_info) > atom_num:
        return None
    for i in atom_info:
        if i.GetSymbol() not in atom_list:
            return None
    return mol


def enumerate_iso(mol):
    return EnumerateStereoisomers(mol)


def disconnect_metal(mol):
    metal_dc = metal.MetalDisconnector()
    return metal_dc(mol)


def charge_optimizer(mol):
    n_tools = normalize.Normalizer()
    return n_tools(mol)


def charge_check(mol):
    un_charge_tool = Uncharger(None)
    return un_charge_tool.uncharge(mol)


class Process(object):
    def __init__(self, mw_sc: tuple = None,
                 re_hydro: bool = None,
                 defn_list='[Na,K,Mg,I,Cl,Br]',
                 atom_list=None,
                 remove_stereo=None,
                 atom_number_limit=5000,
                 shuffle_atoms=False):

        self.remove_stereo = remove_stereo
        self.atom_list = atom_list
        self.defn_list = defn_list
        self.re_hydro = re_hydro
        self.mw_sc = mw_sc
        self.data_length = None
        self.atom_number_limit = atom_number_limit
        self.shuffle = shuffle_atoms
        self.tmp = 0

    def __call__(self, smi_list):
        assert self.atom_list is not None, 'Atom list initiation error!'
        self.data_length = len(smi_list)
        record_return = _map(self.process, smi_list)
        self.tmp = 0
        print()
        return record_return

    def process(self, smi):
        self.tmp += 1
        print('\r' + 'Processing: ' + str(int(self.tmp / self.data_length * 100)) + '%', end='')
        return self._process(smi)

    def _process(self, smi):
        # 输入smiles列表，输出同数量smiles列表，模块主要以mol为处理类型，可以自由替换。
        # 去除混合物
        r_mixture = remove_mixture(smi)

        # 生成mol
        mol = Chem.MolFromSmiles(r_mixture)
        if mol is None:
            return mol

        # 去除无机物
        mol = remove_inorganic(mol)
        if mol is None:
            return mol

        # 去氢/化合物质量过滤
        if self.mw_sc:
            mol = weight_filter(mol, scope=self.mw_sc)
            if mol is None:
                return mol
        elif self.re_hydro:
            mol = add_hydrogen(mol)

        # 断裂金属键
        md = metal.MetalDisconnector()
        mol = md(mol)

        # 中和电荷(不平衡，如N+)
        mol = charge_check(mol)

        # 电荷处理，如S+ O- O-  --->  O=S=O
        mol = charge_optimizer(mol)

        # 去盐
        salt_remover = SaltRemover.SaltRemover(defnData=self.defn_list)
        mol = remove_salt(mol, salt_remover)

        # 原子过滤
        mol = atom_screen(mol, atom_list=self.atom_list, atom_num=self.atom_number_limit)
        if mol is None:
            return mol

        # 价态检查
        Chem.SanitizeMol(mol)

        if not self.shuffle:
            # 是否保持立体异构
            if self.remove_stereo is True:
                return Chem.MolToSmiles(mol, isomericSmiles=False)
            else:
                return Chem.MolToSmiles(mol)
        else:
            mol = shuffle_mol(mol)
            if self.remove_stereo is True:
                return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False)
            else:
                return Chem.MolToSmiles(mol, canonical=False)


def shuffle_mol(mol):
    atom_length = len(mol.GetAtoms())
    atom_index = list(range(atom_length))
    random.shuffle(atom_index)
    r_mol = Chem.RenumberAtoms(mol, atom_index)
    return r_mol

def duplicate_class(func):
    def wrapper(data, data_class):
        if data_class == 'finger':
            print('Fingerprint remove duplicate!')
            return func(data)
        else:
            print('Smiles remove duplicate!')
            return pd.DataFrame(data, columns=['_Smiles'])
    return wrapper


@ duplicate_class
def fp_encoder(smi_list):
    """
    :param smi_list: Valid smiles
    :return:
    """
    # smiles encoder by maccs
    smi = pd.DataFrame(smi_list, columns=['_Smiles'])
    fp_record = list()
    for i in smi_list:
        fp_code = list()
        # fp = np.array(AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(i)))
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i), 2))
        for j in fp:
            fp_code += str(j)
        fp_record += [''.join(fp_code)]
    smi_f = pd.DataFrame(fp_record, columns=['Finger_encoder'])
    return pd.concat((smi, smi_f), axis=1)


if __name__ == '__main__':
    smiles = [
        '[Na]OC(=O)c1ccc(C[S+2]([O-])([O-]))cc1',
        'Br',
        'CCCCCCCCCCCCCCC',
        'CC(C)(Cc1ccccc1)O[Na]',
        'N[S+]([O-])(=O)C1=CC=C(C=C1)C(O)=O',
        'CN[C@@H](C)[C@H](O)C1=CC=CC=C1',
        'NC1=NC2=C(N=CN2COC(CO)CO)C(=O)N1.NC(=N)NC(=O)CC1=C(Cl)C=CC=C1Cl',
        'CCC([O-])(CC)Cc1ccccc1.[Na+]',
        'C=CC(C)(CO)Cc1ccccc1',
        'C=C[C@](C)(CO)Cc1ccccc1',
        'C=C[C@@](C)(CO)Cc1ccccc1'
    ]
    atom_lst = ['C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'Br']
    x = Process(atom_list=atom_lst)
    y = x(smiles)

    # 替换回csv， 先删掉无效的化合物 （或者不替换）
    # 是否使用指纹去重
    # fingerprint编码，返回pd
    # 如为csv，可选择多条件去重， 如为smi，可以直接去重
    y.remove(None)
    # 得到编码的数据
    z = fp_encoder(y, 'finger')
    # 多条件去重
    # ================> 条件:
    print(z)
    z.drop_duplicates(subset=['finger_encoder'], inplace=True)
    print(z)
