from imp import *
from bayes_opt import BayesOptimizer


def bert_main(model_name, model_params, cyp_name, b_epoch):
    internal_epochs = 10
    # train_dataloader, valid_dataloader = torch.load("validation_dataset/Bert_%s.pt" % cyp_name)
    valid_dataloader = torch.load("validation_dataset/evalid_%s.pt" % cyp_name)
    params_ = {k: v for k, v in model_params.items() if (k != "lr") & (k != "pt_epoch")}

    train_loss = AverageMeter()
    valid_loss = AverageMeter()

    train_aucs = Metrics()
    valid_aucs = Metrics()

    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')
    train_results_save, valid_results_save = None, None
    for epoch in range(1):
        model = PredictionModel(**params_)
        model.load_state_dict(torch.load("model_save/MTL_BERT/%s/epoch_%i.pth" % (cyp_name, b_epoch)))
        model = model.to(device)
        count_n = 0
        for i in model.parameters():
            if i.requires_grad:
                count_n += 1
        print("parameters: ", count_n)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["lr"], betas=(0.9, 0.98))
        # train_step(train_dataloader, model, loss_func1, optimizer, train_aucs, train_loss)

        print('epoch: ', epoch, 'train loss: {:.4f}'.format(train_loss.avg))
        # tr_r, tr_c = train_aucs.results()
        # tr_loss = train_loss.avg
        eval_step(valid_dataloader, model, loss_func1, optimizer, valid_aucs, valid_loss)
        print('epoch: ', epoch, 'valid loss: {:.4f}'.format(valid_loss.avg))
        va_r, va_c = valid_aucs.results()
        va_loss = valid_loss.avg

        if epoch == 0:
            valid_results_save = results_combine(va_r, va_c, epoch,
                                                 va_loss, model_name=model_name, data_class="valid")
            # results_save = results_combine(train_results, train_columns, epoch_,
            #                                      train_loss, model_name=model_name, data_class="train")

        else:
            valid_results_save = pd.concat([valid_results_save,
                                            results_combine(va_r, va_c, epoch, va_loss,
                                                            model_name=model_name, data_class="valid")],
                                           axis=0)
    valid_aucs.reset()
    valid_loss.reset()
    train_aucs.reset()
    train_loss.reset()
    return valid_results_save


if __name__ == "__main__":
    pass