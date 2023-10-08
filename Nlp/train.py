import os

from torch.utils.data import DataLoader
from torch import optim
from model import LSTMModel
from model import CNNModel
from main import *
from bayes_opt import BayesOptimizer
from training_for_bert import *

random.seed(main_seed)
torch.manual_seed(main_seed)
torch.cuda.manual_seed(main_seed)
np.random.seed(main_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


model_dict = dict(
    Bi_LSTM=LSTMModel,
    TextCNN=CNNModel,
    MTL_BERT=None
)


def pretraining_main(model_name, model_params):
    print(" -> ")
    with open("data/pad_chem32.pkl", "rb") as file:
        data = pickle.load(file)
    (tr_x, tr_y, tr_w, tr_l), (te_x, te_y, te_w, te_l) = tr_te_split(*data)
    train_data = DataSet(tr_x, tr_y, tr_l, w=tr_w)
    test_data = DataSet(te_x, te_y, te_l, w=te_w)
    train_data_loader = DataLoader(dataset=train_data, batch_size=512, shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=512, shuffle=False)

    model_params = {k: v for k, v in model_params.items() if k != "lr"}
    model_params["class_dim"] = [128]
    model_params = {k: v[0] for k, v in model_params.items()}

    model = model_dict[model_name](**model_params).to(device)
    count_n = 0
    for i in model.parameters():
        if i.requires_grad:
            count_n += 1
    print("parameters: ", count_n)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_f = nn.CrossEntropyLoss(reduction="mean")

    epoch_loss, epoch_acc = [], []
    for epoch in range(10):
        print("epoch:", epoch)
        pretraining_train(model, train_data_loader, optimizer, loss_f, verbose=True)
        batch_acc, batch_loss = pretraining_eval(model, test_data_loader, loss_f, output_smiles=False)
        epoch_loss.append(batch_loss)
        epoch_acc.append(batch_acc)
        if batch_acc >= 0.85:
            torch.save(model.public_model.state_dict(), "pt_model_save/{}/epoch_{}.pth".format(model_name, epoch))

    test_loss = pd.DataFrame(epoch_loss, columns=["test_loss"])
    test_acc = pd.DataFrame(epoch_acc, columns=["test_acc"])
    results_out = pd.concat([test_loss, test_acc], axis=1)
    results_out.to_csv("pt_model_save/{}_pt_loss_results.csv".format(model_name))


def weight_calculation(y, weight):
    weight = [weight[int(i)] for i in y]
    return torch.tensor(weight, dtype=torch.float)


def results_combine(model_results, columns_name, epoch_, model_loss):
    results_save = pd.DataFrame([model_results], columns=columns_name)
    loss_combine = pd.DataFrame([[epoch_, model_loss]], columns=['epoch', 'loss'])
    results_f = pd.concat([loss_combine, results_save], axis=1)
    return results_f


def start(function):
    @wraps(function)
    def wrapper():
        model_list = list(model_dict.keys())
        for model_k, model_name in enumerate(model_list):
            for cyp_name in cyp:
                print("============ model: {} ----- cyp name: {} ============".format(model_name, cyp_name))
                function(cyp_name, model_name)
                print("- END -")
    return wrapper


@start
def start(cyp_name, model_name):
    if model_name != "MTL_BERT":
        # training data loading
        with open("data/{}_train.pkl".format(cyp_name), "rb") as file:
            _, tr_x, tr_y, tr_l = pickle.load(file)

        # testing data loading
        with open("data/{}_test.pkl".format(cyp_name), "rb") as file:
            _, te_x, te_y, te_l = pickle.load(file)

        # valid data loading
        with open("data/{}_valid.pkl".format(cyp_name), "rb") as file:
            _, va_x, va_y, va_l = pickle.load(file)

        # (tr_x, tr_y, tr_l), (va_x, va_y, va_l) = tr_va_split(*train_data_loading, proportions=0.9)
        te_x, te_y, te_l = torch.tensor(te_x, dtype=torch.long), torch.tensor(te_y, dtype=torch.float), torch.tensor(
            te_l, dtype=torch.long)
        train_data = DataSet(tr_x, tr_y, tr_l, data_type="fine")
        valid_data = DataSet(va_x, va_y, va_l, data_type="fine")
        test_data = DataSet(te_x, te_y, te_l, data_type="fine")

        train_data_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
        valid_data_loader = DataLoader(dataset=valid_data, batch_size=128, shuffle=False)
        test_data_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)

        test_data = np.array(tr_y)
        weight = [len(test_data) / (len(test_data) - np.sum(test_data)), len(test_data) / np.sum(test_data)]
        BS = BayesOptimizer(model=model_dict[model_name],
                            model_name=[cyp_name, model_name],
                            params=bayes_params[model_name],
                            random_state=main_seed,
                            epoch=bayes_setting[model_name][0],
                            max_iter=bayes_setting[model_name][1],
                            early_stop_iter=bayes_setting[model_name][2],
                            weight=weight,
                            device=device
                            )

        BS.fit(train_data_loader, valid_data_loader)
        model = BS.best_estimator_.to(device)
        optimizer = BS.best_optimizer_
        b_p = pd.DataFrame([BS.best_params_], columns=None)
        b_p.to_csv("opt_results/{}/{}_params_results.csv".format(cyp_name, model_name), index=False)
        # model = model_init(model)
        # params count
        # model.public_model.load_state_dict(torch.load("pt_model_save/{}/epoch_9.pth".format(model_name)))
        count_n = 0
        for i in model.parameters():
            if i.requires_grad:
                count_n += 1
        print("parameters: ", count_n)
        # model_load
        loss_f = nn.CrossEntropyLoss(reduction="none")
        train_results_save, valid_results_save, test_results_save = None, None, None
        for epoch in range(epochs):
            print("epoch:", epoch)
            train(model, train_data_loader, optimizer, loss_f, weight=weight, verbose=False)
            tr_r, tr_c, tr_loss = eval_(model, train_data_loader, loss_f, weight,
                                        eval_class="train")
            va_r, va_c, va_loss = eval_(model, valid_data_loader, loss_f,
                                        weight=weight, verbose=True)
            te_r, te_c, te_loss = eval_(model, test_data_loader, loss_f,
                                        weight=weight, verbose=True)

            train_results_save, valid_results_save, test_results_save = epoch_results_save(
                epoch, tr_r, tr_c, tr_loss, va_r, va_c, va_loss, te_r, te_c, te_loss,
                train_results_save, valid_results_save, test_results_save
            )

            if va_r[0] > 0.83:
                torch.save(model.state_dict(), "train_model_save/{}/{}/epoch_{}.pth".format(model_name, cyp_name, epoch))
    else:
        train_results_save, valid_results_save, test_results_save = bert_main(main_seed, cyp_name, model_name)

    [i.to_csv("train_results_save/{}/{}_{}_save.csv".format(model_name, cyp_name, j),
              index=False) for i, j in zip([valid_results_save,
                                            train_results_save,
                                            test_results_save],
                                           ('valid',
                                            'train',
                                            'test'))]
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # pretraining_main("Bi_LSTM", bayes_params["Bi_LSTM"])
    # pretraining_main("TextCNN", bayes_params["TextCNN"])
    start()

