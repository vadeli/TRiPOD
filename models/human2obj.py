import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.utils import send_to


class h2o_GAT(nn.Module):
    def __init__(self, cfg, h_dim):
        super(h2o_GAT, self).__init__()
        self.cfg = cfg
        self.h_dim = h_dim
        self.dropout = cfg.model.gat_dropout

        self.attentions = []
        # for i in range(self.cfg.model.gat_layers):
        #     last = 0 if i == 0 else 2 if i == self.cfg.model.gat_layers - 1 else 1
        self.attentions = [GraphAttentionLayer(cfg, h_dim, last=0, concat=True) for _ in range(cfg.model.h2o_gat_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(cfg, h_dim, last=3, concat=False)

        # self.attentions = [GraphAttentionLayer(h_dim, h_dim, dropout=self.dropout, alpha=cfg.model.gat_alpha, concat=True) for _ in range(cfg.model.h2o_gat_heads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        # self.out_att = GraphAttentionLayer(h_dim * cfg.model.h2o_gat_heads, nclass, dropout=self.dropout, alpha=cfg.model.gat_alpha, concat=False)


    def forward(self, h_states, o_states, seq_start_end_p, seq_start_end_o):

        batch_h2o_h = []
        batch = 0
        for _, (start, end) in enumerate(seq_start_end_p):
            # start, end = start.item(), end.item()
            curr_hidden_human = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_obj = o_states.view(-1, self.h_dim)[seq_start_end_o[batch][0]:seq_start_end_o[batch][1]]
            curr_hidden = torch.cat((curr_hidden_human, curr_hidden_obj), dim=0)
            batch += 1

            adj = torch.zeros([curr_hidden.shape[0], curr_hidden.shape[0]])
            adj[curr_hidden_human.shape[0]:, :curr_hidden_human.shape[0]] = 1
            adj[:curr_hidden_human.shape[0], curr_hidden_human.shape[0]:] = 1
            indent = torch.eye(curr_hidden.shape[0])
            indent[curr_hidden_human.shape[0]:, curr_hidden_human.shape[0]:] = 0
            adj = torch.add(adj, indent)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            adj = send_to(adj, device)

            x = F.dropout(curr_hidden, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            x = F.log_softmax(x, dim=1)

            batch_h2o_h.append(x[:curr_hidden_human.shape[0],:])
        batch_h2o_h = torch.cat(batch_h2o_h, dim=0)
        return batch_h2o_h


class GraphAttentionLayer(nn.Module):
    def __init__(self, cfg, h_dim, last=0, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.cfg = cfg
        self.last = last
        self.h_dim = h_dim
        self.concat = concat
        self.alpha = cfg.model.gat_alpha

        if self.last == 3:
            embed_dim = self.cfg.model.h2o_gat_heads * self.h_dim
            bottleneck_dim = self.h_dim
            self.in_features = embed_dim
            self.out_features = bottleneck_dim
        elif self.last == 1 or self.last == 2:
            embed_dim = self.cfg.model.h2o_gat_heads * self.h_dim
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

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [12,12]

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.cfg.model.gat_dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
