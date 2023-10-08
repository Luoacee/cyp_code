from main import *
from bayes_opt import BayesOptimizer
from torch import optim
def bert_main(seed, cyp_name, model_name, dir_, bayes_choice=True):
    ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''
    model_scale = medium

    num_layers = model_scale['num_layers']
    num_heads = model_scale['num_heads']
    d_model = model_scale['d_model']

    dff = d_model * 4

    seed = seed
    np.random.seed(seed=seed)

    train_data = pd.read_csv("split/{}/Data_In_MACCS/cyp{}_train_MACCS.csv".format(dir_, cyp_name)).loc[:, ["Smiles", "Y"]]
    valid_data = pd.read_csv("data/{}/Data_In_MACCS/cyp{}_valid_MACCS.csv".format(dir_, cyp_name)).loc[:, ["Smiles", "Y"]]
    test_data = pd.read_csv("data/{}/Data_In_MACCS/cyp{}_test_MACCS.csv".format(dir_, cyp_name)).loc[:, ["Smiles", "Y"]]
    columns = ["Smiles", "Y"]
    # train_df, valid_df = split_train_valid_for_bert(train_data)
    test_df = test_data
    train_df = train_data
    valid_df = valid_data

    train_dataset = Prediction_Dataset(train_df, smiles_head=smiles_head,
                                       reg_heads=reg_heads, clf_heads=clf_heads)
    test_dataset = Prediction_Dataset(test_df, smiles_head=smiles_head,
                                      reg_heads=reg_heads, clf_heads=clf_heads)
    valid_dataset = Prediction_Dataset(valid_df, smiles_head=smiles_head,
                                       reg_heads=reg_heads, clf_heads=clf_heads)

    kwargs = {"clf_heads": clf_heads, "reg_heads": reg_heads}
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=Finetune_Collater(kwargs))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=Finetune_Collater(kwargs))
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=Finetune_Collater(kwargs))
    # torch.save(test_dataloader, "validation_dataset/evalid_%s.pt" % cyp_name)
    # return None, None ,None
    if bayes_choice:
        # BS = BayesOptimizer(model=PredictionModel,
        #                     model_name=[cyp_name, model_name],
        #                     params=bayes_params[model_name],
        #                     random_state=main_seed,
        #                     epoch=bayes_setting[model_name][0],
        #                     max_iter=bayes_setting[model_name][1],
        #                     early_stop_iter=bayes_setting[model_name][2],
        #                     device=device
        #                     )
        #
        # BS.fit(train_dataloader, valid_dataloader)
        # model = BS.best_estimator_.to(device)
        # optimizer = BS.best_optimizer_
        # b_p = pd.DataFrame([BS.best_params_], columns=None)
        # b_p.to_csv("opt_results/{}/{}_params_results.csv".format(cyp_name, model_name), index=False)
        load_params = pd.read_csv("opt_results/{}/{}_params_results.csv".format(cyp_name, model_name)).to_dict()
        params_ = {k: v[0] for k, v in load_params.items() if k != "lr" and k != "pt_epoch"}
        model = PredictionModel(**params_).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=load_params["lr"][0], betas=(0.9, 0.98))

    else:
        # x, property = next(iter(train_dataset))
        model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads,
                                vocab_size=bert_vocab_size,
                                dropout_rate=0.1, reg_nums=len(kwargs["reg_heads"]), clf_nums=len(kwargs["clf_heads"]))
        model.encoder.load_state_dict(torch.load('pt_model_save/MTL_BERT/medium_weights_bert_encoder_weightsmedium_20.pt'))
    model = model.to(device)
    count_n = 0


    # if pretraining:
    #     model.encoder.load_state_dict(torch.load())
    #     print('load_wieghts')

    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-4, betas=(0.9, 0.98))

    train_loss = AverageMeter()
    test_loss = AverageMeter()
    valid_loss = AverageMeter()

    train_aucs = Metrics()
    test_aucs = Metrics()
    valid_aucs = Metrics()

    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')

    stopping_monitor = 0

    train_results_save, valid_results_save, test_results_save = None, None, None
    for epoch in range(b_epoch):
        train_step(train_dataloader, model, loss_func1, optimizer, train_aucs, train_loss)

        print('epoch: ', epoch, 'train loss: {:.4f}'.format(train_loss.avg))
        tr_r, tr_c = train_aucs.results()
        tr_loss = train_loss.avg
        eval_step(valid_dataloader, model, loss_func1, optimizer, valid_aucs, valid_loss)
        print('epoch: ', epoch, 'valid loss: {:.4f}'.format(valid_loss.avg))
        va_r, va_c = valid_aucs.results()
        va_loss = valid_loss.avg
        eval_step(test_dataloader, model, loss_func1, optimizer, test_aucs, test_loss)
        print('epoch: ', epoch, 'test loss: {:.4f}'.format(test_loss.avg))
        te_r, te_c = test_aucs.results()
        te_loss = test_loss.avg
        train_results_save, valid_results_save, test_results_save = epoch_results_save(
            epoch, tr_r, tr_c, tr_loss, va_r, va_c, va_loss, te_r, te_c, te_loss,
            train_results_save, valid_results_save, test_results_save
        )
        # if va_r[0] >= 0.86:
        #     torch.save(model.state_dict(), "train_model_save/MTL_BERT/{}/epoch_{}.pth".format(cyp_name, epoch))

        valid_aucs.reset()
        valid_loss.reset()
        train_aucs.reset()
        train_loss.reset()
        test_aucs.reset()
        test_loss.reset()
        print()
    return train_results_save, valid_results_save, test_results_save


if __name__ == "__main__":
    pass