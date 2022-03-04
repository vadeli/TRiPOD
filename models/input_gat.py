import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.utils import send_to


class inp_GAT(nn.Module):
    def __init__(self, cfg, h_dim, node_hsize):
        super(inp_GAT, self).__init__()
        self.cfg = cfg
        self.h_dim = h_dim
        self.dropout = cfg.model.gat_dropout
        self.node_hsize = node_hsize

        self.attentions = []
        # for i in range(self.cfg.model.gat_layers):
        #     last = 0 if i == 0 else 2 if i == self.cfg.model.gat_layers - 1 else 1
        self.attentions = [GraphAttentionLayer(cfg, h_dim, node_hsize, last=0, concat=True) for _ in range(cfg.model.gat_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(cfg, h_dim, node_hsize, last=3, concat=False)

        # self.attentions = [GraphAttentionLayer(h_dim, h_dim, dropout=self.dropout, alpha=cfg.model.gat_alpha, concat=True) for _ in range(cfg.model.gat_heads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        # self.out_att = GraphAttentionLayer(h_dim * cfg.model.gat_heads, nclass, dropout=self.dropout, alpha=cfg.model.gat_alpha, concat=False)


    def forward(self, inputs, seq_start_end):

        in_frames = inputs.shape[0]
        person_num = inputs.shape[1]
        joints = int(inputs.shape[2] / self.node_hsize)
        dim = self.cfg.dataset.dim

        social_h_tmp = []

        for f in range(in_frames):
            social_h = []
            for p in range(person_num):
                cur_framep = inputs[f][p]
                offset = inputs[f][p][0:joints * dim].view(-1, dim)
                graph = offset
                if self.node_hsize > dim:
                    loc = inputs[f][p][joints * dim: joints * (dim*2)].view(-1, dim)
                    graph = torch.cat((graph, loc), dim=1)
                    if self.node_hsize > dim*2:
                        S = inputs[f][p][joints * (dim*2):].view(-1, 1)
                        graph = torch.cat((graph, S), dim=1)
                x = F.dropout(graph, self.dropout, training=self.training)
                if self.cfg.model.input_Graph_type == "FC":
                    adj = 0
                elif self.cfg.model.input_Graph_type == "sparse":
                    adj = torch.zeros([x.shape[0], x.shape[0]])
                    for b in range(len(self.cfg.dataset.bones)):
                        curbone = self.cfg.dataset.bones[b]
                        adj[curbone[0], curbone[1]] = 1
                        adj[curbone[1], curbone[0]] = 1
                        adj[b, b] = 1
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    adj = send_to(adj, device)

                x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.elu(self.out_att(x, adj))
                x = F.log_softmax(x, dim=1)

                social_h.append(x.view(x.size(0) * x.size(1)))
            social_h_tmp.append(torch.stack(social_h))
        social_h_tmp = torch.stack(social_h_tmp)

        return social_h_tmp



class GraphAttentionLayer(nn.Module):
    def __init__(self, cfg, h_dim, node_hsize, last=0, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.cfg = cfg
        self.last = last
        self.h_dim = h_dim
        self.concat = concat
        self.alpha = cfg.model.gat_alpha
        self.node_hsize = node_hsize

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
            self.in_features = self.node_hsize
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

        if self.cfg.model.input_Graph_type == "sparse":
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)  # [12,12]
        else:
            attention = e

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.cfg.model.gat_dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
