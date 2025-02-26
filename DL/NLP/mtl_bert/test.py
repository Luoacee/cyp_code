import torch
import sys
def make_trg_mask(trg,trg_pad_idx=0):
    trg_pad_mask = (trg == trg_pad_idx).unsqueeze(1).unsqueeze(3)
    print(trg_pad_mask.shape)

    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
    print(trg_sub_mask.shape)

    trg_mask = trg_pad_mask & trg_sub_mask
    print(trg_mask.shape)
    sys.exit(1)
    return trg_mask



trg = torch.ones(64, 110)
make_trg_mask(trg)
