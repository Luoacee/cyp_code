import torch
from seq_encoding import *

device = ["cuda" if torch.cuda.is_available() else "cpu"][0]
main_seed = 100
epochs = 200
cyp = [
    "2c9",
    "2d6",
    "3a4"
]

bayes_params = dict(
    Bi_LSTM=dict(
        vocab_size=[len(c2i.keys())],
        input_dim=[200],
        hidden_dim=[512],
        lr=[0.001, 0.0005, 0.0001],
        model_dim=[512],
        class_dim=[128, 256, 300, 512]

    ),
    TextCNN=dict(
        vocab_size=[len(c2i.keys())],
        input_dim=[200],
        hidden_dim=[512],
        lr=[0.001, 0.0005, 0.0001],
        head=[60],
        kernel_size=[[1, 3, 5]],
        class_dim=[128, 256, 300, 512]

    ),
    MTL_BERT=dict(
        num_layers=[6],
        d_model=[300],
        dff=[300*4],
        num_heads=[6],
        vocab_size=[70],
        dropout_rate=[0.1, 0.2],
        lr=[0.001, 0.0005, 0.0001],
        reg_nums=[0],
        clf_nums=[1],
        pt_epoch=[20, 60, 100]
    )
)

bayes_setting = dict(
    Bi_LSTM=(130, 10, 5),
    TextCNN=(130, 10, 5),
    MTL_BERT=(130, 12, 5)
)
# bayes_setting = dict(
#     Bi_LSTM=(2, 1, 8),
#     TextCNN=(2, 1, 8),
#     MTL_BERT=(20, 1, 8)
# )

# --------------------- bert params
b_epoch = 200
smiles_head = ["Smiles"]
clf_heads = ["Y"]
reg_heads = []
# small = {'name':'Small','num_layers': 4, 'num_heads': 4, 'd_model': 128,'path':'small_weights'}
medium = {'name': 'medium', 'num_layers': 6, 'num_heads': 6, 'd_model': 300, 'path': 'medium_weights'}
# large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 512,'path':'large_weights'}
bert_vocab_size = 70
dropout_rate = 0.1
