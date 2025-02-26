import torch
import torch.nn as nn
from model_params import *


class NNBase(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, output_dim, model_dim=None, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.dropout = dropout
        self.emb = nn.Embedding(vocab_size, input_dim)
        self.output_dim = output_dim

        self.public_seq = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.2)
        )


class LSTMBase(NNBase):
    def __init__(self, vocab_size, input_dim, model_dim, hidden_dim, bidirectional=2, dropout=0.2):
        super().__init__(vocab_size=vocab_size,
                         input_dim=input_dim,
                         model_dim=model_dim,
                         hidden_dim=hidden_dim,
                         dropout=dropout,
                         output_dim=model_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.model_dim,
            batch_first=True,
            num_layers=1,
            bias=True,
            bidirectional=bool(bidirectional),
        )
        self.ln = nn.LayerNorm(self.model_dim)

    def params_initiation(self, x):
        c = torch.zeros(size=[2, x.shape[0], self.model_dim]).to(device)
        h = torch.zeros(size=[2, x.shape[0], self.model_dim]).to(device)
        return c, h

    def forward(self, x, lth):
        batch_size = x.shape[0]
        x = self.emb(x)
        c, h = self.params_initiation(x)
        out, (c, h) = self.lstm(x, (c, h))
        f_dim = []
        for idx, i in enumerate(lth):
            f_dim += [out[idx, i - 1, :].unsqueeze(0)]
        x = torch.concat(f_dim, dim=0)
        x = x[:, :self.model_dim] + x[:, self.model_dim:]
        return self.public_seq(self.ln(x))


class CNNBlock(nn.Module):
    def __init__(self, intput_head, output_head, kernel_size, kernel_dim):
        super().__init__()
        self.cnn1 = nn.Conv2d(intput_head, output_head, kernel_size=(kernel_size[0], kernel_dim))
        self.cnn2 = nn.Conv2d(intput_head, output_head, kernel_size=(kernel_size[1], kernel_dim))
        self.cnn3 = nn.Conv2d(intput_head, output_head, kernel_size=(kernel_size[2], kernel_dim))

    def forward(self, x):
        return self.cnn1(x), self.cnn2(x), self.cnn3(x)



class CNNBase(NNBase):
    def __init__(self, vocab_size, input_dim, hidden_dim, head, kernel_size, dropout=0.2):
        super().__init__(vocab_size=vocab_size,
                         input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         dropout=dropout,
                         output_dim=head * len(kernel_size) * 2)
        self.head = head
        self.kernel_size = kernel_size
        self.cnn1 = CNNBlock(1, self.head, kernel_size=kernel_size, kernel_dim=input_dim)
        self.cnn2 = CNNBlock(self.head, self.head*2, kernel_size=kernel_size, kernel_dim=1)

        self.max_p = nn.MaxPool2d((self.calculate(150, kernel_size) - kernel_size[0] + 1 , 1))
        self.max_p2 = nn.MaxPool2d((self.calculate(150, kernel_size) - kernel_size[1] + 1 , 1))
        self.max_p3 = nn.MaxPool2d((self.calculate(150, kernel_size) - kernel_size[2] + 1 , 1))
        self.relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(self.head)
        self.ln = nn.LayerNorm(len(kernel_size) * head * 2)

    def forward(self, x):
        x = self.cnn_part(x)
        x = self.public_seq(x)
        return x

    @staticmethod
    def calculate(x, kernel):
        return x * 4 - sum(kernel) + 3

    def cnn_part(self, x):
        x_c = self.emb(x)
        x_c = x_c.unsqueeze(1)
        x_c_1, x_c_2, x_c_3 = self.cnn1(x_c)

        x = x.unsqueeze(1).unsqueeze(-1).expand(-1, self.head, -1, -1)
        x_c = torch.concat([x_c_1,x_c_2, x_c_3, x], dim=-2)
        x_c = self.relu(self.bn(x_c))
        x_c2_1, x_c2_2, x_c2_3 = self.cnn2(x_c)

        x_c11 = self.max_p(x_c2_1)
        x_c21 = self.max_p2(x_c2_2)
        x_c31 = self.max_p3(x_c2_3)
        x_cf = torch.concat([x_c11, x_c21, x_c31], dim=1)
        x_cf = x_cf.squeeze(-1).squeeze(-1)
        x_cf = self.relu(self.ln(x_cf))
        return x_cf


class PretrainingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.o_emb = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
            bias=True,
            num_layers=1
        )
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.vocab_size),
            nn.LayerNorm(self.vocab_size),
            nn.Dropout(0.2)
        )

        self.relu = nn.LeakyReLU(0.2)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, output):
        batch_size, seq_length = x.shape
        x = self.o_emb(x)
        h0 = output.unsqueeze(0)
        c0 = output.unsqueeze(0)
        ot, (_, _) = self.lstm(x, (h0, c0))
        ot = self.output(ot)
        return ot


class ClassificationModel(nn.Module):
    def __init__(self, hidden_dim, class_dim, class_n):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.class_n = class_n
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, int(class_dim*1.5)),
            nn.LayerNorm(int(class_dim*1.5)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(class_dim*1.5), class_dim),
            nn.LayerNorm(class_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(class_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.output(x)

    def lstm(self, x, lth=None, dim_select=None):
        if dim_select == "All":
            x = x[:, -1, :]
            return x
        else:
            f_dim = []
            for idx, i in enumerate(lth):
                f_dim += [x[idx, i - 1, :].unsqueeze(0)]
            x = torch.concat(f_dim, dim=0)
            return x


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, input_dim, model_dim, hidden_dim, class_dim, class_n=2, dropout=0.2):
        super().__init__()
        self.public_model = LSTMBase(vocab_size=vocab_size,
                                     input_dim=input_dim,
                                     model_dim=model_dim,
                                     hidden_dim=hidden_dim,
                                     dropout=dropout)
        self.pt_block = PretrainingModel(hidden_dim=hidden_dim, vocab_size=vocab_size, input_dim=input_dim)
        self.ft_block = ClassificationModel(hidden_dim, class_dim, class_n)

    def classification(self, x, lth):
        x = self.public_model(x, lth)
        return self.ft_block(x)

    def pretraining(self, input_data, lth):
        x = self.public_model(input_data, lth)
        return self.pt_block(input_data, x)


class CNNModel(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, kernel_size, head, class_dim, class_n=2, dropout=0.2):
        super().__init__()
        print("Init CNN model")
        self.public_model = CNNBase(vocab_size=vocab_size,
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    head=head,
                                    kernel_size=kernel_size,
                                    dropout=dropout)
        self.pt_block = PretrainingModel(hidden_dim=hidden_dim, vocab_size=vocab_size, input_dim=200)
        self.ft_block = ClassificationModel(hidden_dim, class_dim, class_n)


    def classification(self, x, lth):
        x = self.public_model(x)
        return self.ft_block(x)

    def pretraining(self, input_data, lth):
        x = self.public_model(input_data)
        return self.pt_block(input_data, x)
