import torch
import warnings
import dgl
import torch_geometric.nn as tn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
import torch.nn as nn
import dgl.nn.pytorch as dn
from dgllife.model.model_zoo import AttentiveFPPredictor, GATPredictor
from torch_geometric.utils import scatter

devices = ["cuda" if torch.cuda.is_available() else "cpu"][0]
warnings.filterwarnings("ignore", category=UserWarning, module="dgl")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class GNN(nn.Module):
    def __init__(self, hidden_dim, out_dim, class_number, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.class_number = class_number
        self.ln = nn.LayerNorm(hidden_dim)
        self.seq = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(out_dim // 2, class_number),
            nn.Softmax(-1)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(hidden_dim, 768),
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2),
            nn.Linear(768, class_number),
            nn.Softmax(-1)
        )
        self.seq3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_number)
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, *args, **kwargs):
        return None


class AttentiveFP(GNN):
    def __init__(self, atom_dim, bond_dim, hidden_dim, out_dim, class_number, dropout=0.2):
        super().__init__(hidden_dim=hidden_dim,
                         out_dim=out_dim,
                         class_number=class_number,
                         dropout=dropout)
        self.gnn_layer = AttentiveFPGNN(node_feat_size=atom_dim,
                                        edge_feat_size=bond_dim,
                                        graph_feat_size=hidden_dim)
        self.att_layer = AttentiveFPReadout(feat_size=hidden_dim, num_timesteps=2, dropout=dropout)

    def forward(self, data, get_node_weight=True):
        x = data[0].ndata["features"]
        edge_x = data[0].edata["features"]
        x = self.gnn_layer(data[0], x, torch.as_tensor(edge_x, dtype=torch.float32))
        if get_node_weight:
            x, aw = self.att_layer(data[0], x, get_node_weight=get_node_weight)
            x = self.seq2(x)
            return x, aw
        else:
            x = self.att_layer(data[0], x, get_node_weight=get_node_weight)
            x = self.seq2(x)
            return x
    @staticmethod
    def split_graph(data_edge, data_batch):
        batch_index = data_batch
        graph_list = []
        temp_node = 0
        src_edge_index = [int(tmp) for tmp in data_edge[0]]
        dec_edge_index = [int(tmp) for tmp in data_edge[1]]
        for i in range(max(batch_index)+1):
            src, dec = [], []
            node_n = sum([int(k) for k in batch_index == i])
            edges_range = list(range(temp_node, temp_node+node_n))
            mk = 0
            for j, k in zip(src_edge_index, dec_edge_index):
                if j in edges_range:
                    src.append(j)
                    assert k in edges_range, "edge not in node"
                    dec.append(k)
                    mk = 1
                else:
                    if mk == 1:
                        break
                    else:
                        continue

            src_edge_index = src_edge_index[len(src):]
            dec_edge_index = dec_edge_index[len(dec):]
            src = [int(te - temp_node) for te in src]
            dec = [int(te - temp_node) for te in dec]
            assert min(src) == 0 and max(src) == node_n-1, "graph error"
            graph_list.append(dgl.graph((src, dec)))
            temp_node += node_n
        g = dgl.batch(graph_list)
        return g


class GCN(GNN):
    def __init__(self, atom_dim, hidden_dim, out_dim, class_number, dropout=0.2):
        super().__init__(hidden_dim=hidden_dim,
                         out_dim=out_dim,
                         class_number=class_number,
                         dropout=dropout)
        self.GCN1 = tn.GCNConv(atom_dim, hidden_dim, improved=True)
        self.GCN2 = tn.GCNConv(hidden_dim, hidden_dim)

    def forward(self, data, get_node_weight=True):
        x = data.x
        edges = data.edge_index
        x = self.GCN1(x, edges)
        x = self.relu(self.ln(x))
        x = self.GCN2(x, edges)
        x = self.relu(x)
        torch.use_deterministic_algorithms(False)
        x = global_max_pool(x, data.batch)
        torch.use_deterministic_algorithms(True)
        x = self.seq2(x)
        return x


class GATpy(GNN):
    def __init__(self, atom_dim, hidden_dim, out_dim, class_number, head_n, dropout=0.2):
        super().__init__(hidden_dim=hidden_dim,
                         out_dim=out_dim,
                         class_number=class_number,
                         dropout=dropout)

        self.GAT1 = tn.GATConv(in_channels=atom_dim, out_channels=hidden_dim, heads=head_n, concat=False)
        self.GAT2 = tn.GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=head_n)
        self.seq = nn.Sequential(
            nn.Linear(hidden_dim*head_n, hidden_dim),
            self.seq
        )

    def forward(self, data, return_attention_weights=True):
        x = data.x
        edges = data.edge_index
        x = self.GAT1(x, edges)
        x = self.relu(self.ln(x))
        if return_attention_weights:
            x, a = self.GAT2(x, edges, return_attention_weights=return_attention_weights)
            x = self.relu(x)
            x = global_mean_pool(x, data.batch)
            x = self.seq(x)
            return x, a
        else:
            x = self.GAT2(x, edges, return_attention_weights=return_attention_weights)
            x = self.relu(x)
            x = global_mean_pool(x, data.batch)
            x = self.seq(x)
            return x


class GATdgl(GNN):
    def __init__(self, atom_dim, hidden_dim, out_dim, class_number, head_n, dropout=0.2):
        super().__init__(hidden_dim=hidden_dim,
                         out_dim=out_dim,
                         class_number=class_number,
                         dropout=dropout)

        self.gnn_layer = dn.GATConv(in_feats=atom_dim, out_feats=hidden_dim, num_heads=head_n)
        self.att_layer = dn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=head_n)
        self.seq = nn.Sequential(
            nn.Linear(hidden_dim*head_n, hidden_dim),
            self.seq2
        )

    def forward(self, data, return_attention_weights=True):
        g = data[0]
        x = g.ndata["features"]
        x = self.gnn_layer(g, x)
        x = torch.mean(x, dim=1)
        x = self.relu(self.ln(x))
        x, aw = self.att_layer(g, x, get_attention=return_attention_weights)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        with g.local_scope():
            g.ndata['x'] = x
            x = dgl.max_nodes(g, 'x')
        x = self.seq(x)
        return x, aw


class GIN(GNN):
    def __init__(self, atom_dim, hidden_dim, out_dim, class_number, dropout=0.2):
        super().__init__(hidden_dim=hidden_dim,
                         out_dim=out_dim,
                         class_number=class_number,
                         dropout=dropout)
        self.GIN1 = tn.GIN(in_channels=atom_dim, num_layers=1, hidden_channels=hidden_dim, out_channels=hidden_dim)
        self.GIN2 = tn.GIN(in_channels=hidden_dim, num_layers=1, hidden_channels=hidden_dim, out_channels=hidden_dim)

    def forward(self, data, get_node_weight=True):
        x = data.x
        edges = data.edge_index
        x = self.GIN1(x, edges)
        x = self.relu(self.ln(x))
        x = self.GIN2(x, edges)
        x = self.relu(x)
        # torch.use_deterministic_algorithms(False)
        x = global_max_pool(x, data.batch)
        # torch.use_deterministic_algorithms(True)
        x = self.seq2(x)
        return x


class GATPr(nn.Module):
    def __init__(self, atom_dim,hidden_dim, head_n, class_number, out_dim, dropout=0.2):
        super().__init__()
        self.prt = GATPredictor(in_feats=atom_dim, hidden_feats=[hidden_dim, hidden_dim], num_heads=[head_n, head_n], n_tasks=class_number)

    def forward(self, data, get_node_weight=True):
        x = data[0].ndata["features"]
        x = self.prt(data[0], x)
        return x, None
