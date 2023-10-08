import os
from functools import wraps
from bayes_opt import *

# Random seed
random.seed(main_seed)
os.environ['PYTHONHASHSEED'] = str(main_seed)
torch.manual_seed(main_seed)
torch.cuda.manual_seed_all(main_seed)
torch.cuda.manual_seed(main_seed)
np.random.seed(main_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

# Some params
model_dict = dict(
    AttentiveFP=AttentiveFP,
    GCN=GCN,
    GAT=GATPr,
    GIN=GIN
)


def main(function):
    @wraps(function)
    def wrapper():
        model_list = list(model_dict.keys())
        for model_k, model_name in enumerate(model_list):
            for cyp_name in cyp:
                print("============ model: {} ----- cyp name: {} ============".format(model_name, cyp_name))
                if "AT".lower() in model_name.lower():
                    attention_method = True
                    p_method = "DGL"
                else:
                    attention_method = False
                    p_method = "PYG"
                function(model_name, cyp_name, p_method, attention_method, model_init_method[model_k])
                print("- END -")
    return wrapper


@main
def main(model_name, cyp_name, process_method, attn, init_method, bayes_opt=True):
    train_path = "processing_save/cyp{}_train.pt".format(cyp_name)
    test_path = "processing_save/cyp{}_test.pt".format(cyp_name)
    valid_path = "processing_save/cyp{}_valid.pt".format(cyp_name)
    tr_data, te_data, va_data, weight = data_pipline(train_path=train_path,
                                                     valid_path=valid_path,
                                                     test_path=test_path, method=process_method
                                                     )
    torch.save(va_data, "validation_dataset/no_edge_info/%s_valid_data.pt" % cyp_name)
    return None
    loss_function = nn.CrossEntropyLoss(reduction="none")
    if bayes_opt:
        BS = BayesOptimizer(model=model_dict[model_name],
                            model_name=[cyp_name, model_name],
                            params=bayes_params[model_name],
                            random_state=main_seed,
                            epoch=bayes_setting[model_name][0],
                            max_iter=bayes_setting[model_name][1],
                            early_stop_iter=bayes_setting[model_name][2],
                            weight=weight,
                            device=devices
                            )

        BS.fit(tr_data, va_data, attn)
        model = BS.best_estimator_.to(devices)
        optimizer = BS.best_optimizer_
        b_p = pd.DataFrame([BS.best_params_], columns=None)
        b_p.to_csv("opt_results/{}/{}_params_results.csv".format(cyp_name, model_name), index=False)
    else:
        model = model_dict[model_name](**model_params[model_name])
        optimizer = optim.Adam(model.parameters(),
                               model_lr)
    train_results_save, valid_results_save, test_results_save = None, None, None
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        train(model, tr_data, optimizer, loss_function, weight, attn=attn)

        train_results, train_columns, train_loss = eval_(model, tr_data, loss_function,
                                                         weight, attn=attn, eval_class="train")
        valid_results, valid_columns, valid_loss = eval_(model, va_data, loss_function, weight, attn=attn)
        test_results, test_columns, test_loss = eval_(model, te_data, loss_function, weight, attn=attn)

        if epoch == 0:
            valid_results_save = results_combine(valid_results, valid_columns, epoch, valid_loss)
            train_results_save = results_combine(train_results, train_columns, epoch, train_loss)
            test_results_save = results_combine(test_results, test_columns, epoch, test_loss)
        else:
            valid_results_save = pd.concat([valid_results_save,
                                            results_combine(valid_results, valid_columns, epoch, valid_loss)], axis=0)
            train_results_save = pd.concat([train_results_save,
                                            results_combine(train_results, train_columns, epoch, train_loss)], axis=0)
            test_results_save = pd.concat([test_results_save,
                                           results_combine(test_results, test_columns, epoch, test_loss)], axis=0)
        if valid_results[0] > 0.85:
            torch.save(model.state_dict(), "model_save/{}/{}/epoch_{}.pth".format(cyp_name, model_name, epoch))

    [i.to_csv("results_save/{}/{}_{}_save.csv".format(model_name, cyp_name, j),
              index=False) for i, j in zip([valid_results_save,
                                            train_results_save,
                                            test_results_save],
                                           ('valid',
                                            'train',
                                            'test'))]


if __name__ == "__main__":
    main()
