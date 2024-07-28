import torch.nn as nn
import torch


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    aa = torch.fft.rfft2(a, dim=(-2, -1))
    aaa = torch.stack((aa.real, aa.imag), -1)
    bb = torch.fft.rfft2(b, dim=(-2, -1))
    bbb = torch.stack((bb.real, bb.imag), -1)
    m = com_mult(conj(aaa), bbb)
    # ifft 仅包含实部 irfft包含实部和虚部
    out = torch.fft.irfft2(torch.complex(m[..., 0], m[..., 1]), a.size())
    # cor_m = aa * bb.conj()
    # correlation = torch.fft.irfft2(cor_m, a.size())
    return out


class RelGCNConv(nn.Module):
    def __init__(self, in_features, out_features, num_rel):
        super(RelGCNConv, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features, bias=True) for i in range(num_rel)])
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, adjacency_list):
        x_list = []
        for i, adjacency_hat in enumerate(adjacency_list):
            x_l = self.linears[i](x)
            x_d = torch.mm(adjacency_hat, x_l)
            x_list.append(x_d.unsqueeze(0))
        x = self.tanh(torch.sum(torch.cat(x_list, dim=0), dim=0))
        return x


class RelGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_rel, num_layer):
        super().__init__()
        layers = []
        if num_layer == 1:
            layers.append(RelGCNConv(in_features, out_features, num_rel))
        else:
            for i in range(num_layer):
                if i == 0:
                    layers.append(RelGCNConv(in_features, hidden_features, num_rel))
                elif i == num_layer - 1:
                    layers.append(RelGCNConv(hidden_features, out_features, num_rel))
                else:
                    layers.append(RelGCNConv(hidden_features, hidden_features, num_rel))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adjacency_list):
        for layer in self.layers:
            x = layer(x, adjacency_list)
        return x


class CompGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_rels, dropout=0.1, opn='corr'):
        super(self.__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = nn.Tanh()
        self.node_linears = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=True) for _ in range(self.num_rels)])
        self.rel_linears = nn.ModuleList(
            [nn.Linear(1, out_channels, bias=True) for _ in range(self.num_rels)])

        self.w_loop = nn.Linear(self.in_channels, self.out_channels)
        self.w_in = nn.Linear(self.in_channels, self.out_channels)
        self.w_out = nn.Linear(self.in_channels, self.out_channels)
        self.w_rel = nn.Linear(self.in_channels, self.out_channels)
        self.loop_rel = nn.Parameter(torch.Tensor(1, self.in_channels))
        self.drop = torch.nn.Dropout(p=dropout)
        # self.norm = self.compute_norm(self.in_index, num_ent)
        # corr sub mult
        self.opn = opn
        self.bias = 0
        if self.bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))

    def forward(self, node_embed, rel_embed, adj_list, adj_lambda):
        out = node_embed
        for i, adj in enumerate(adj_list):
            n_embed = self.node_linears[i](node_embed)
            r_embed = self.rel_linears[i](adj)
            # 进行rel_transform操作
            res = self.message(ent_embed=n_embed, rel_embed=r_embed, edge_norm=None)
            res = self.drop(res)
            # out = out + res * (1 / len(adj_list))
            out = out + res * (adj_lambda[i] / sum(adj_lambda))
        if self.bias:
            out = out + self.bias
        # out = self.bn(out)
        out = self.act(out)
        return out

    def message(self, ent_embed, rel_embed, edge_norm):
        ent_embed = ent_embed.float()
        rel_embed = rel_embed.float()
        if self.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed if edge_norm is None else trans_embed * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class CompGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_rels, conv_num=1, dropout=0.1, opn='corr'):
        super().__init__()
        self.dropout = dropout
        self.opn = opn
        self.conv_num = conv_num
        self.num_rels = num_rels
        self.conv1 = CompGCNConv(in_channels, out_channels, self.num_rels, self.dropout, self.opn)
        self.conv2 = None
        self.conv3 = None
        if conv_num > 1:
            self.conv2 = CompGCNConv(in_channels, out_channels, self.num_rels, self.dropout, self.opn)
        if conv_num > 2:
            self.conv3 = CompGCNConv(in_channels, out_channels, self.num_rels, self.dropout, self.opn)

    def forward(self, node_embed: torch.Tensor, h_embed: torch.Tensor, t_embed: torch.Tensor, rel_embed: torch.Tensor,
                adjacency_list, adj_lambda):
        n = torch.cat([h_embed, t_embed], dim=0)
        for i, adj in enumerate(adjacency_list):
            adjacency_list[i] = torch.cat([adj, adj], dim=0)
        r = torch.cat([rel_embed, rel_embed], dim=0)
        n = self.conv1(n, r, adjacency_list, adj_lambda)
        if self.conv_num > 1:
            n = self.conv2(n, r, adjacency_list)
        if self.conv_num > 2:
            n = self.conv3(n, r, adjacency_list)
        return n
