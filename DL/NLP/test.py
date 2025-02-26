import functools
import os
from model import LSTMModel
from model import CNNModel
import torch
from test_for_bert import bert_main
from bayes_opt import *
# best_epoch ->
# no random state ->
devices = "cuda" if torch.cuda.is_available() else "cpu"
model_dict = dict(
    Bi_LSTM=LSTMModel,
    TextCNN=CNNModel,
    MTL_BERT=None,
)
# 0
epoch_choices = dict(
    MTL_BERT=[8, 4, 154]
)
# -1
epoch_choices = dict(
    Bi_LSTM=[48, 50, 44],
    MTL_BERT=[9, 4, 5],
    TextCNN=[45, 302, 28]
)

cyp_names = [
    "2c9",
    "2d6",
    "3a4"
]


def valid_p(func):
    def wrapper():
        for m in model_dict.keys():
            for c_index, c in enumerate(cyp_names):
                print(m, c)
                model_params_load = pd.read_csv("opt_results/%s/%s_params_results.csv" % (c, m)).to_dict()
                _model_params = {k: v[0] for k, v in model_params_load.items()}
                b_epoch = epoch_choices[m][c_index]
                func(m, _model_params, c, b_epoch)
                torch.cuda.empty_cache()
    return wrapper


@valid_p
def valid_p(model_name, model_params_, cy_name, b_epoch):
    internal_epochs = 10
    if model_name != "MTL_BERT":
        # data_loading_hear
        # -> torch.load
        print("validation_dataset/%s_validation_data.pt" % cy_name)
        train_data, valid_data, weight = torch.load("validation_dataset/normal_%s.pt" % cy_name)
        # optimizer & loss_f
        loss_function = nn.CrossEntropyLoss(reduction="none")
        model_head = model_dict[model_name]

        params_ = {k: v for k, v in model_params_.items() if k != "lr"}
        try:
            kernel_size = eval(params_["kernel_size"])
            params_["kernel_size"] = kernel_size
        except:
            pass
        valid_results_save, train_results_save = None, None
        for epoch_ in range(internal_epochs):
            # internal import
            model = model_head(**params_).to(devices)

            optimizer = torch.optim.Adam(model.parameters(), lr=model_params_["lr"])
            # print("model_load->", cyp_name, " ", model_name, " ", b_epoch, )
            model.load_state_dict(torch.load("model_save/%s/%s/epoch_%s.pth" % (model_name, cy_name,  b_epoch)))
            for _ in range(1):
                train(model, train_data, optimizer, loss_function, weight=weight)
            # valid_results, valid_columns, valid_loss = eval_(model, train_data, loss_function,
            #                                                  weight, attn=attn, eval_class="train")
            valid_results, valid_columns, valid_loss = eval_(model, valid_data, loss_function, weight)

            if epoch_ == 0:
                valid_results_save = results_combine(valid_results, valid_columns, epoch_,
                                                     valid_loss, model_name=model_name, data_class="valid")
                # results_save = results_combine(train_results, train_columns, epoch_,
                #                                      train_loss, model_name=model_name, data_class="train")

            else:
                valid_results_save = pd.concat([valid_results_save,
                                                results_combine(valid_results, valid_columns, epoch_, valid_loss,
                                                                model_name=model_name, data_class="valid")],
                                               axis=0)
                # results_save = pd.concat([results_save,
                #                                 results_combine(train_results, train_columns,
                #                                                 epoch_, train_loss, model_name=model_name, data_class="train")],
                #                                axis=0)
    else:
        valid_results_save = bert_main(model_name, model_params_, cy_name, b_epoch)
    [i.to_csv("10_random_validation/{}_{}_{}_save.csv".format(model_name, cy_name, j),
              index=False) for i, j in zip([valid_results_save
                                            ],
                                           (['exvalid']
                                            ))]



if __name__ == "__main__":
    valid_p()


