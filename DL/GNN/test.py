import functools
import os

import torch
from bayes_opt import *
# best_epoch ->
# no random state ->
model_dict = dict(
    AttentiveFP=AttentiveFP,
    GCN=GCN,
    GAT=GATdgl,
    GIN=GIN
)
epoch_choices = dict(
    AttentiveFP=[422, 289, 361],
    GCN=[498, 407, 495],
    GAT=[34, 389, 466],
    GIN=[358, 184, 473]
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
                if "AT".lower() in m.lower():
                    attention_method = True
                else:
                    attention_method = False
                b_epoch = epoch_choices[m][c_index]
                func(m, _model_params, c, b_epoch, attention_method)
    return wrapper


@valid_p
def valid_p(model_name, model_params_, cyp_name, b_epoch, attn):
    internal_epochs = 1
    # data_loading_hear
    # -> torch.load
    if attn:
        d_path = "with_edge_info"
    else:
        d_path = "no_edge_info"
    print("validation_dataset/%s/%s_evalid_data.pt" % (d_path, cyp_name))
    valid_data= torch.load("validation_dataset/%s/%s_evalid_data.pt" % (d_path, cyp_name))
    # optimizer & loss_f
    loss_function = nn.CrossEntropyLoss(reduction="none")
    model_head = model_dict[model_name]

    params_ = {k: v for k, v in model_params_.items() if k != "lr"}
    valid_results_save, train_results_save = None, None
    for epoch_ in range(internal_epochs):
        # internal import
        model = model_head(**params_).to(devices)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_params_["lr"])
        # print("model_load->", cyp_name, " ", model_name, " ", b_epoch, )
        model.load_state_dict(torch.load("model_save/%s/%s/epoch_%s.pth" % (cyp_name, model_name, b_epoch)))
        # train(model, train_data, optimizer, loss_function, weight=weight, attn=attn)
        # valid_results, valid_columns, valid_loss = eval_(model, train_data, loss_function,
        #                                                  weight, attn=attn, eval_class="train")
        weight=[1, 1]
        valid_results, valid_columns, valid_loss = eval_(model, valid_data, loss_function, weight, attn=attn)

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

    [i.to_csv("10_random_validation/{}_{}_{}_save.csv".format(model_name, cyp_name, j),
              index=False) for i, j in zip([valid_results_save
                                            ],
                                           ('evalid',
                                            ))]


if __name__ == "__main__":
    valid_p()


