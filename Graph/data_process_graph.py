import os
# from graph_gen import gen_plot
# from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem
from rdkit.Chem import SanitizeMol
import pandas as pd
import numpy as np
from model_params import graph_atoms
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
import torch
import copy
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType
from networkx import Graph
from tqdm import tqdm

max_p_length = 42

# 读取数据
# 数据编码
# 编写成DATA的图形式，图的形式方便后期训练，可以获得邻接矩阵，节点张量


def _map(*args):
    return list(map(args[0], args[1]))


def csv_read(path):
    return pd.read_csv(path)


def remove_invalid_data(data, smiles_column_name='Smiles'):
    read_smiles = list(data[smiles_column_name])

    def _check_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        else:
            SanitizeMol(mol)
            return Chem.MolToSmiles(mol)

    normal_smiles = _map(lambda x: _check_smiles(x), read_smiles)
    assert len(normal_smiles) >= 0, "No valid smiles!"
    data[smiles_column_name] = normal_smiles
    data.dropna(subset=["Smiles"], inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data


class MolecularEncoder(object):
    """
    encoder_unknown: True False or GCNN_paper
    encoder_vector: DGL or GCNN_paper
    """

    # get feature vectors
    def __init__(self, atom_list=None, encoder_unknown=True, encoder_vector=None):
        self.encoder_unknown = encoder_unknown
        if encoder_vector == 'DGL':
            self.mk = 1
        else:
            self.mk = 0

        # symbol
        self.atom_list = atom_list

        # degree
        self.degree = list(range(6))
        self.dgl_degree = list(range(11))

        # explicit valence
        self.explicit_valence = list(range(7))
        self.dgl_explicit_valence = list(range(7))

        # formal_charge
        self.dgl_formal_charge_range = list(range(-2, 3))
        self.formal_charge_range = list(range(-1, 2))

        # hybridization
        self.hybridization = [HybridizationType.SP,
                              HybridizationType.SP2,
                              HybridizationType.SP3,
                              ]
        self.dgl_hybridization = [HybridizationType.SP,
                                  HybridizationType.SP2,
                                  HybridizationType.SP3,
                                  HybridizationType.SP3D,
                                  HybridizationType.SP3D2
                                  ]

        # IsAromatic
        # use_chirality

        # radical electrons
        self.dgl_radical_electrons_range = list(range(5))

        # total num H
        self.dgl_total_num_H = list(range(5))
        self.total_num_H = list(range(5))

        # =========================> bond_info
        self.bond_type = [BondType.SINGLE,
                          BondType.DOUBLE,
                          BondType.TRIPLE,
                          BondType.AROMATIC]

        # Bond conjugated
        # Bond is in ring

        self.bond_chirality = [
            "STEREONONE",
            "STEREOANY",
            "STEREOZ",
            "EOE"
        ]


    # 获取节点特征信息, 编码成onehot张量

    def __call__(self, smiles):
        # 扔一个smiles
        # 得到编码的维度
        get_mol = Chem.MolFromSmiles(smiles)
        n_atom, x, b = self.process_mol_info(get_mol) # 返回原子个数
        edges_connect = self.get_bond(get_mol)
        return x, b, edges_connect, n_atom

    @staticmethod
    def one_hot_encoder(atom, match_set, encoder_unknown=False):
        # list of class
        temp_list = copy.deepcopy(match_set)
        if encoder_unknown:
            temp_list.append("<UNK>")
            if atom in temp_list:
                return [atom == a for a in temp_list]
            else:
                atom = "<UNK>"
                return [atom == a for a in temp_list]
        # other encoder
        else:
            return [atom == a for a in temp_list]

    def get_chirality(self, atom, encoder_unknown=False):
        # get chirality prop  [R, S] + exit(0, 1)
        c_pos = atom.HasProp('_ChiralityPossible')
        try:
            atom_c = self.get_mol_info(atom.GetProp('_CIPCode'), ['R', 'S'], encoder_unknown=encoder_unknown)
        except KeyError:
            if encoder_unknown:
                atom_c = [0, 0, 0]
            else:
                atom_c = [0, 0]
        atom_c = list(atom_c)
        atom_c.append(c_pos)
        return np.array(atom_c)

    def process_mol_info(self, mol):
        mk = self.mk
        full_a_features = list()
        full_b_features = list()
        atom_i = defaultdict(list)
        bond_i = defaultdict(list)
        get_atom = mol.GetAtoms()
        get_bond = mol.GetBonds()
        atom_i['Symbol'] = _map(lambda x: self.get_mol_info(x.GetSymbol(),
                                                             self.atom_list,
                                                             encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Degree'] = _map(lambda x: self.get_mol_info(x.GetDegree(),
                                                             [self.degree, self.dgl_degree][mk],
                                                             encoder_unknown=False), get_atom)
        atom_i['FormalCharge'] = _map(lambda x: self.get_mol_info(x.GetFormalCharge(),
                                                                   [self.formal_charge_range,
                                                                    self.dgl_formal_charge_range][mk],
                                                                   encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Explicit_valence'] = _map(lambda x: self.get_mol_info(x.GetExplicitValence(),
                                                                       [self.explicit_valence,
                                                                        self.dgl_explicit_valence][mk],
                                                                       encoder_unknown=False), get_atom)
        atom_i['Hybridization'] = _map(lambda x: self.get_mol_info(x.GetHybridization(),
                                                                    [self.hybridization, self.dgl_hybridization][mk],
                                                                    encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Total_num_H'] = _map(lambda x: self.get_mol_info(x.GetTotalNumHs(),
                                                                  [self.total_num_H, self.dgl_total_num_H][mk],
                                                                  encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Aromatic'] = _map(lambda x: np.array([x.GetIsAromatic()]), get_atom)

        if mk:
            # DGL
            atom_i['Radical_electrons'] = _map(lambda x: self.get_mol_info(x.GetNumRadicalElectrons(),
                                                                            self.dgl_radical_electrons_range,
                                                                            encoder_unknown=self.encoder_unknown),
                                               get_atom)
            atom_i['Chirality'] = _map(lambda x: self.get_chirality(x, encoder_unknown=self.encoder_unknown),
                                       get_atom)
        else:
            atom_i['Radical_electrons'] = _map(lambda x: self.get_mol_info(x.GetNumRadicalElectrons(),
                                                                            self.dgl_radical_electrons_range,
                                                                            encoder_unknown=self.encoder_unknown),
                                               get_atom)
            atom_i['Chirality'] = _map(lambda x: self.get_chirality(x, encoder_unknown=self.encoder_unknown),
                                       get_atom)

        bond_i["Bond_type"] = _map(lambda x: self.get_mol_info(x.GetBondType(),
                                                                  self.bond_type,
                                                                  encoder_unknown=self.encoder_unknown), get_bond)
        bond_i["Stereo"] = _map(lambda x: self.get_mol_info(x.GetStereo(),
                                                               self.bond_chirality,
                                                               encoder_unknown=self.encoder_unknown), get_bond)
        bond_i["Is_in_ring"] = _map(lambda x: np.array([x.IsInRing()]), get_bond)
        bond_i["Conjugate"] = _map(lambda x: np.array([x.GetIsConjugated()]), get_bond)
        atom_f_keys = list(atom_i.keys())
        bond_f_keys = list(bond_i.keys())

        for k in atom_f_keys:
            full_a_features += [atom_i[k]]
        full_a_features = np.concatenate(full_a_features, axis=1)
        record_atom_features, record_bond_features = [], []
        for i in full_a_features:
            record_atom_features += [i/i.sum()]

        for b in bond_f_keys:
            full_b_features += [bond_i[b]]
        full_b_features = np.concatenate(full_b_features, axis=1)
        for j in full_b_features:
            record_bond_features += [j/j.sum()]

        return mol.GetNumAtoms(), record_atom_features, record_bond_features

    def get_mol_info(self, atom, match_set, encoder_unknown=None):
        # Get prop matrix from atom
        return np.array(self.one_hot_encoder(atom, match_set=match_set, encoder_unknown=encoder_unknown))

    # 获取边信息
    @ staticmethod
    def get_bond(mol):
        bonds_info = mol.GetBonds()
        bond_s = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds_info]

        G = Graph(bond_s).to_directed()

        get_b = np.array(G.edges).T
        return get_b


def _pipline(file_path, return_label=True):
    file_read = csv_read(file_path)
    normal_data = remove_invalid_data(file_read)
    smiles = normal_data["Smiles"]
    x = []
    M = MolecularEncoder(atom_list=graph_atoms)
    # 返回值为SMILES的原子特征，键特征以及原子的个数
    print('Smiles encoding....')
    for i in tqdm(smiles):
        x += [M(i)]
    if return_label:
        return [smiles, x] + [file_read["Y"].values.tolist()]
    else:
        return [smiles, x]

# def protein_encoder(protein_seq_list):
#     print('Protein encoding....')
#     encoder_list = list()
#     for i in tqdm(protein_seq_list):
#         temp_list = []
#         for index, j in enumerate(i):
#             if index >= max_p_length:
#                 break
#             else:
#                 temp_list.append(protein_dict[j])
#         encoder_list += [np.array(temp_list)]
#     return np.array(encoder_list)


def save_pt(_data, _slice, file_name):
    torch.save((_data, _slice), file_name)


def gen_plot_data(file_path, save_name=None):
    k = _pipline(file_path)
    _data, _slice = gen_plot(k)
    save_pt(_data, _slice, save_name)


def gen_plot(data):
    graph_record = []
    for smiles, graph_info, label in zip(*data):

        x_feature = graph_info[0]
        edge_feature = graph_info[1]
        edge_info = graph_info[2]
        atom_number = graph_info[3]
        graph_info = Data(x=torch.asarray(np.array(x_feature), dtype=torch.float32),
                          edge_feature=torch.asarray(np.array(edge_feature)), dtype=torch.float32,
                          edge_index=torch.tensor(edge_info, dtype=torch.long),
                          y=torch.tensor(label, dtype=torch.long),
                          smiles=smiles,
                          smiles_size=torch.tensor(atom_number, dtype=torch.long)
                          )
        graph_record.append(graph_info)
    inm = InMemoryDataset()
    _data, _slice = inm.collate(graph_record)
    return _data, _slice


if __name__ == '__main__':
    # for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     cyp_full = ['2c9', '2d6', '3a4']
    #     for cyp_name in cyp_full:
    #         data_file_train = 'split/{}/Dataset_In_RDKIT/cyp{}_train_RDKIT.csv'.format(i, cyp_name)
    #         data_file_test = 'split/{}/Dataset_In_RDKIT/cyp{}_test_RDKIT.csv'.format(i, cyp_name)
    #         data_file_valid = 'split/{}/Dataset_In_RDKIT/cyp{}_valid_RDKIT.csv'.format(i, cyp_name)
    #
    #         gen_plot_data(data_file_train, save_name='split/{}/cyp{}_train.pt'.format(i, cyp_name))
    #         gen_plot_data(data_file_test, save_name='split/{}/cyp{}_test.pt'.format(i, cyp_name))
    #         gen_plot_data(data_file_valid, save_name='split/{}/cyp{}_valid.pt'.format(i, cyp_name))
    import os
    cyp_full = ['2c9', '2d6', '3a4']
    for cyp_name in cyp_full:
        for k in os.listdir("domain/RDKIT/cyp%s" % cyp_name):
            print(k)
            k_ = "domain/RDKIT/cyp%s/%s" % (cyp_name, k)
            gen_plot_data(k_, save_name='processing_save/domain_data/cyp%s/%s.pt' % (cyp_name, k.rstrip(".csv")))
