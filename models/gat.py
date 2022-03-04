import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, cfg, h_dim):
        super(GAT, self).__init__()
        self.cfg = cfg
        self.h_dim = h_dim
        self.dropout = cfg.model.gat_dropout

        self.attentions = []
        # for i in range(self.cfg.model.gat_layers):
        #     last = 0 if i == 0 else 2 if i == self.cfg.model.gat_layers - 1 else 1
        self.attentions = [GraphAttentionLayer(cfg, h_dim, last=0, concat=True) for _ in range(cfg.model.gat_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(cfg, h_dim, last=3, concat=False)


    def forward(self, h_states, seq_start_end_p):

        social_h = []
        for _, (start, end) in enumerate(seq_start_end_p):
            # start, end = start.item(), end.item()
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]

            x = F.dropout(curr_hidden, self.dropout, training=self.training)
            x = torch.cat([att(x) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x))

            social_h.append(F.log_softmax(x, dim=1))
        social_h = torch.cat(social_h, dim=0)
        return social_h


class GraphAttentionLayer(nn.Module):
    def __init__(self, cfg, h_dim, last=0, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.cfg = cfg
        self.last = last
        self.h_dim = h_dim
        self.concat = concat
        self.alpha = cfg.model.gat_alpha

        if self.last == 3:
            embed_dim = self.cfg.model.gat_heads * self.h_dim
            bottleneck_dim = self.h_dim
            self.in_features = embed_dim
            self.out_features = bottleneck_dim
        elif self.last == 1 or self.last == 2:
            embed_dim = self.cfg.model.gat_heads * self.h_dim
            self.in_features = embed_dim
            self.out_features = self.h_dim
        else:
            embed_dim = self.h_dim
            self.in_features = self.h_dim
            self.out_features = embed_dim

        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def forward(self, input):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        attention = e

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.cfg.model.gat_dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
