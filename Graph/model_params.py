import torch

graph_atoms = [
    'C', 'O', 'N', 'P', 'S', 'Si', 'Cl', 'Br', 'I', 'F'
]
main_seed = 200
cyp = ["2c9", "2d6", "3a4"]
# cyp = ["2d6"]
model_init_method = ["kaiming", "kaiming", "kaiming", "kaiming"]
lr_decay = False
model_lr = None
epochs = 500
atom_dim = 49
bond_dim = 12
p_dropout = 0.2
model_params = dict(
    AttentiveFP=dict(atom_dim=49, bond_dim=12, fingerprint_dim=200, out_dim=128, class_number=2),
    GCN=dict(atom_dim=49, hidden_dim=512, out_dim=256, class_number=2),
    GAT=dict(atom_dim=49, hidden_dim=512, head_n=6, out_dim=256, class_number=2),
    GIN=dict(atom_dim=49, hidden_dim=512, out_dim=256, class_number=2)
)
devices = ["cuda" if torch.cuda.is_available() else "cpu"][0]
weight_decay = 2.9
learning_rate = 3.5

bayes_params = dict(
    AttentiveFP=dict(atom_dim=[atom_dim],
                     bond_dim=[bond_dim],
                     hidden_dim=[100, 150, 200, 250, 512],
                     out_dim=[256, 128, 64],
                     class_number=[2],
                     lr=[0.005, 0.001, 0.0005],
                     dropout=[p_dropout]
                     ),
    GCN=dict(atom_dim=[atom_dim],
             hidden_dim=[128, 256, 300, 512],
             out_dim=[256, 128],
             class_number=[2],
             lr=[0.005, 0.001, 0.0005],
             dropout=[p_dropout]
             ),
    GAT=dict(atom_dim=[atom_dim],
             hidden_dim=[128, 256, 300, 512],
             out_dim=[256, 128, 64],
             class_number=[2],
             head_n=[3, 6, 8],
             lr=[0.005, 0.001, 0.0005],
             dropout=[p_dropout]
             ),
    GIN=dict(atom_dim=[atom_dim],
             hidden_dim=[128, 256, 300, 512],
             out_dim=[256, 128],
             class_number=[2],
             lr=[0.005, 0.001, 0.0005],
             dropout=[p_dropout]
             ),

)

bayes_setting = dict(
    AttentiveFP=(200, 35, 6),
    GCN=(150, 20, 8),
    GAT=(200, 60, 6),
    GIN=(150, 20, 8)
)
# bayes_setting = dict(
#     AttentiveFP=(2, 1, 6),
#     GCN=(2, 1, 6),
#     GAT=(2, 1, 6),
#     GIN=(2, 1, 6)
# )

opt_params_path = "opt_results"