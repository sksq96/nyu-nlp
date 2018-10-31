import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()
        self.params = params
        self.conv1 = nn.Conv1d(
            params.d_embed, params.d_hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            params.d_hidden, params.d_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input is: (length, batch_size, num_channels)
        # conv1d module expects: (batch_size, num_channels, length)
        h0 = x.transpose(0, 1).transpose(1, 2).contiguous()

        h0 = self.relu(self.conv1(h0))
        h0 = self.relu(self.conv2(h0))

        # return (batch_size, num_channels)
        h0 = h0.transpose(1, 2)
        return torch.sum(h0, dim=1)


class RNNEncoder(nn.Module):
    def __init__(self, params):
        super(RNNEncoder, self).__init__()
        self.params = params
        input_size = params.d_embed
        dropout = 0 if params.n_layers == 1 else params.dp_ratio
        self.rnn = nn.GRU(input_size=input_size, hidden_size=params.d_hidden,
                        num_layers=params.n_layers, dropout=dropout,
                        bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.params.n_cells, batch_size, self.params.d_hidden
        h0 =  inputs.new_zeros(state_shape)
        _, ht = self.rnn(inputs, h0)

        # bring batch_size to the 0th dim
        ht = ht[-2:].transpose(0, 1).contiguous()
        # concat forward and backward rnn hidden
        return ht.view(batch_size, -1)


class NLI(nn.Module):

    def __init__(self, params):
        super(NLI, self).__init__()
        
        self.params = params
        self.embed = nn.Embedding(params.n_embed, params.d_embed)
        if params.encoder == 'rnn':
            self.encoder = RNNEncoder(params)
        elif params.encoder == 'cnn':
            self.encoder = CNNEncoder(params)
        else:
            raise ValueError(f'Encoder {params.encoder} is not supported. Try using cnn or rnn.')

        
        self.dropout = nn.Dropout(p=params.dp_ratio)
        self.relu = nn.ReLU()
        
        fc_in_size = params.d_hidden
        # concat s1 and s2
        fc_in_size *= 2
        if params.encoder == 'rnn':
            # concat forward and backward bi-rnn
            fc_in_size *= 2

        fc_ot_size = params.d_fc
        
        # 2-layers fc
        self.out = nn.Sequential(
            nn.Linear(fc_in_size, fc_ot_size),
            self.relu,
            self.dropout,
            nn.Linear(fc_ot_size, params.d_out)
        )

    def forward(self, s1, s2):
        # fix embeddings, do not backprop
        s1_embed = self.embed(s1).detach()
        s2_embed = self.embed(s2).detach()
        
        s1_encode = self.encoder(s1_embed)
        s2_encode = self.encoder(s2_embed)
        
        return self.out(torch.cat([s1_encode, s2_encode], 1))

