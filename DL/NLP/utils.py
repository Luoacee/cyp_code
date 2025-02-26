from model_params import *
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset
random.seed(main_seed)
torch.manual_seed(main_seed)
torch.cuda.manual_seed(main_seed)
np.random.seed(main_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def tr_te_split(*args, proportions: float = 0.95) -> Union[tuple, tuple]:
    data_index = list(range(len(args[0])))
    random.shuffle(data_index)
    train_index = data_index[:int(len(data_index) * proportions)]
    test_index = data_index[int(len(data_index) * proportions):]
    train_x, train_y, train_w, train_l = args[0][train_index], args[1][train_index], \
        args[2][train_index], args[3][train_index]
    test_x, test_y, test_w, test_l = args[0][test_index], args[1][test_index], args[2][test_index], args[3][test_index]
    return (torch.tensor(train_x, dtype=torch.long),
            torch.tensor(train_y, dtype=torch.long),
            torch.tensor(train_w, dtype=torch.float),
            torch.tensor(train_l, dtype=torch.long)), \
        (torch.tensor(test_x, dtype=torch.long),
         torch.tensor(test_y, dtype=torch.long),
         torch.tensor(test_w, dtype=torch.float),
         torch.tensor(test_l, dtype=torch.long))


class DataNode(object):
    def __init__(self, smiles=None, x=None, y=None, length=None):
        self.smiles = smiles
        self.x = x
        self.y = y
        self.length = length

    def __repr__(self):
        return "{} -> smiles={}, x={}, y={}, length={}".format(self.__class__.__name__,
                                                               self.smiles, self.x, self.y,
                                                               self.length)


def tr_va_split(*args, proportions: float = 0.90) -> Union[tuple, tuple]:
    all_train_data = [DataNode(smiles=args[0][i],
                               x=args[1][i],
                               y=args[2][i],
                               length=args[3][i]) for i in range(len(args[0]))]

    train_data, test_data = train_test_split(all_train_data, shuffle=True,
                                             random_state=main_seed, train_size=proportions, stratify=args[2])

    return (torch.tensor(np.array([train_data[i].x for i in range(len(train_data))]), dtype=torch.long),
            torch.tensor(np.array([train_data[i].y for i in range(len(train_data))]), dtype=torch.float),
            torch.tensor(np.array([train_data[i].length for i in range(len(train_data))]), dtype=torch.long)), \
        (torch.tensor(np.array([test_data[i].x for i in range(len(test_data))]), dtype=torch.long),
         torch.tensor(np.array([test_data[i].y for i in range(len(test_data))]), dtype=torch.float),
         torch.tensor(np.array([test_data[i].length for i in range(len(test_data))]), dtype=torch.long))


class DataSet(Dataset):
    def __init__(self, x, y, lth, w=None, data_type="pretraining"):
        super().__init__()
        self.xdata = x
        self.ydata = y
        self.wdata = w
        self.lth_data = lth
        self.data_type = data_type

    def __getitem__(self, item):
        if self.data_type == "pretraining":
            return self.xdata[item], self.ydata[item], self.wdata[item], self.lth_data[item]
        else:
            return self.xdata[item], self.ydata[item], self.lth_data[item]

    def __len__(self):
        return len(self.xdata)


def model_init(model, method="xavier"):
    init_method = method
    for name, w in model.named_parameters():
        if "embedding" not in name:
            if 'weight' in name:
                if len(w.shape) >= 2:
                    if init_method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif init_method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
    return model


def results_combine(model_results, columns_name, epoch_, model_loss, model_name=None, data_class=None):
    results_save = pd.DataFrame([model_results], columns=columns_name)
    if model_name is None:
        loss_combine = pd.DataFrame([[epoch_, model_loss]], columns=['epoch', 'loss'])
    else:
        loss_combine = pd.DataFrame([[model_name, data_class, epoch_, model_loss]], columns=['model', "class", 'epoch', 'loss'])
    results_f = pd.concat([loss_combine, results_save], axis=1)
    return results_f


def epoch_results_save(epoch, tr_r, tr_c, tr_loss,
                       va_r, va_c, va_loss,
                       te_r, te_c, te_loss,
                       train_results_save,
                       valid_results_save,
                       test_results_save):
    if epoch == 0:
        valid_results_save = results_combine(va_r, va_c, epoch, va_loss)
        train_results_save = results_combine(tr_r, tr_c, epoch, tr_loss)
        test_results_save = results_combine(te_r, te_c, epoch, te_loss)
    else:
        valid_results_save = pd.concat([valid_results_save,
                                        results_combine(va_r, va_c, epoch, va_loss)], axis=0)
        train_results_save = pd.concat([train_results_save,
                                        results_combine(tr_r, tr_c, epoch, tr_loss)], axis=0)
        test_results_save = pd.concat([test_results_save,
                                       results_combine(te_r, te_c, epoch, te_loss)], axis=0)
    return train_results_save, valid_results_save, test_results_save


def split_train_valid_for_bert(train_data):
    smiles, y = train_data.Smiles.values, train_data.Y.values
    train_x, valid_x, train_y, valid_y = train_test_split(list(smiles), list(y),
                                        random_state=100, shuffle=True, train_size=0.9,
                                        stratify=y)
    return pd.DataFrame(np.array([train_x, train_y]).T, columns=["Smiles", "Y"]), pd.DataFrame(np.array([valid_x, valid_y]).T, columns=["Smiles", "Y"])
