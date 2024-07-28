import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from .loss_func import categorical_crossentropy


def convert_table_to_node(cr_table: torch.Tensor = None, re_table: torch.Tensor = None, span_len=None):
    offset = 0
    cr_adj, re_adj = [], []
    for l in span_len:
        cr_sub = cr_table[offset: offset + l * l].view(l, l)
        re_sub = re_table[offset: offset + l * l].view(l, l)
        cr_sub = torch.softmax(cr_sub, dim=-1)
        re_sub = torch.softmax(re_sub, dim=-1)
        cr_adj.append(cr_sub)
        re_adj.append(re_sub)
        offset += l * l
    cr_adj = torch.block_diag(*cr_adj)
    re_adj = torch.block_diag(*re_adj)
    return cr_adj, re_adj


def convert_table_to_node_2(cr_table: torch.Tensor = None, re_table: torch.Tensor = None, span_len=None):
    offset = 0
    cr_adj, re_adj = [], []
    type_num = re_table.shape[1]
    for l in span_len:
        cr_sub = cr_table[offset: offset + l * l].view(l, l)
        re_sub = re_table[offset: offset + l * l].view(l, l, type_num)
        cr_sub = torch.softmax(cr_sub, dim=-1)
        re_sub = torch.softmax(re_sub, dim=-1)
        cr_adj.append(cr_sub)
        re_adj.append(re_sub)
        offset += l * l
    cr_adj = torch.block_diag(*cr_adj)
    re_adj_list = []
    re_adj_type_num_list = [[] for _ in range(type_num)]
    # 遍历batch中的每个tensor
    for tensor in re_adj:
        # 遍历每个块，并将其添加到对应的列表中
        for i in range(type_num):
            block = tensor[:, :, i]
            re_adj_type_num_list[i].append(block)
    for _ in re_adj_type_num_list:
        re_adj_list.append(torch.block_diag(*_))
    re_adj = torch.stack(re_adj_list, dim=2)
    return cr_adj, re_adj


def convert_graph_to_table(graph: torch.Tensor = None, span_len=None):
    offset = 0
    table = []
    for l in span_len:
        x = graph[offset: offset + l, offset: offset + l]
        x = x.reshape(l * l, -1)
        table.append(x)
        offset += l
    table = torch.cat(table, dim=0)
    return table


def convert_node_to_table(nodes: torch.Tensor = None, span_len=None):
    offset = 0
    hss, tss = [], []
    for l in span_len:
        x = nodes[offset: offset + l]
        hs = x.repeat(1, l).view(l * l, -1)
        ts = x.repeat(l, 1)
        hss.append(hs)
        tss.append(ts)
        offset += l
    hss = torch.cat(hss, dim=0)
    tss = torch.cat(tss, dim=0)
    return hss, tss


def form_table_input(sequence_output: torch.Tensor = None, span_pos=None, strategy='max-pooling'):
    """
    input:          span_pos is a [batch[span pos[]]]
                    hts is a [batch[ht pair]]
    return a tuple: node_embeds: tensor, node_len: list
         satisfied: \sum{node_len} = len(node_embed)

    """
    span_len = [len(row) for row in span_pos]
    span_embed = []
    for i, row in enumerate(span_pos):
        for span in row:
            emb = sequence_output[i, span[0]:span[1], :]
            if strategy == 'max-pooling':
                emb = torch.max(emb, dim=0)[0]
                span_embed.append(emb)
            if strategy == 'marker':
                emb = emb[0]
                span_embed.append(emb)
            else:
                raise ValueError("Unimplemented strategy.")
    span_embed = torch.stack(span_embed, dim=0)
    return span_embed, span_len


def get_hrt(sequence_output: torch.Tensor = None, attention: torch.Tensor = None,
            batch_span_pos=None, batch_hts=None, type_sequence_output=None, type_input_list=None):
    """
    logsumexp pooling，最大池化的平滑版本，得到实体嵌入hei，修改版
    span_pos and hts are in batch format.
    attention shape: (batch_size, num_heads, sequence_length, sequence_length).
    """
    hss, tss, rss, span_embss, span_attss, hs_type_embss, ts_type_embss = [], [], [], [], [], [], []
    for i, (span_pos, hts, type_input_ids) in enumerate(zip(batch_span_pos, batch_hts, type_input_list)):
        span_embs, span_atts, type_embs, hs_type_embs, ts_type_embs = [], [], [], [], []
        for idx, span in enumerate(span_pos):
            emb = sequence_output[i, span[0]]
            att = attention[i, :, span[0]]
            span_embs.append(emb)
            span_atts.append(att)
            type_embs.append(type_sequence_output[type_input_list[i][idx]])
            span_embss.append(emb)
            span_attss.append(att)
        span_embs = torch.stack(span_embs, dim=0)  # [n_s, d]
        span_atts = torch.stack(span_atts, dim=0)  # [n_s, num_heads, seq_len]
        type_embs = torch.stack(type_embs, dim=0)

        hts = torch.LongTensor(hts).to(sequence_output.device)
        hs = torch.index_select(span_embs, 0, hts[:, 0])
        ts = torch.index_select(span_embs, 0, hts[:, 1])

        hs_type_embs = torch.index_select(type_embs, 0, hts[:, 0])
        ts_type_embs = torch.index_select(type_embs, 0, hts[:, 1])

        h_att = torch.index_select(span_atts, 0, hts[:, 0])
        t_att = torch.index_select(span_atts, 0, hts[:, 1])
        ht_att = (h_att * t_att).mean(1)
        ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
        rs = contract("ld,rl->rd", sequence_output[i], ht_att)
        hss.append(hs)
        tss.append(ts)
        rss.append(rs)
        hs_type_embss.append(hs_type_embs)
        ts_type_embss.append(ts_type_embs)
    hss = torch.cat(hss, dim=0)
    tss = torch.cat(tss, dim=0)
    rss = torch.cat(rss, dim=0)
    span_embss = torch.stack(span_embss, dim=0)
    span_attss = torch.stack(span_attss, dim=0)
    hs_type_embss = torch.cat(hs_type_embss, dim=0)
    ts_type_embss = torch.cat(ts_type_embss, dim=0)
    return hss, tss, rss, span_embss, span_attss, hs_type_embss, ts_type_embss


def gen_hts(span_len: int):
    return [[i, j] for i in range(span_len) for j in range(span_len)]


class BaseModel(nn.Module):
    def __init__(self, hidden_dim: int, emb_dim: int, block_dim: int, num_class: int, sample_rate: float,
                 lossf: nn.Module, dataset=str, dropout=0.1):
        super().__init__()
        # validation: input can be chunked into blocks
        assert emb_dim % block_dim == 0, "emb_dim must be multiple of block_dim."
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.emb_dim = emb_dim
        self.block_dim = block_dim
        self.num_block = hidden_dim // block_dim
        self.num_class = num_class
        self.dropout = nn.Dropout(p=dropout)
        self.head_extractor = nn.Linear(2 * hidden_dim + 256, emb_dim)
        self.tail_extractor = nn.Linear(2 * hidden_dim + 256, emb_dim)
        # origin
        self.head_ext = nn.Linear(2 * hidden_dim, emb_dim)
        self.tail_ext = nn.Linear(2 * hidden_dim, emb_dim)
        self.head_type_extra = nn.Linear(hidden_dim, hidden_dim)
        self.tail_type_extra = nn.Linear(hidden_dim, hidden_dim)
        self.head_type_extractor = nn.Linear(hidden_dim, 256)
        self.tail_type_extractor = nn.Linear(hidden_dim, 256)
        self.linear = nn.Linear(2 * emb_dim, num_class, bias=False)
        self.bilinear = nn.Linear(emb_dim * block_dim, num_class)  # size: [(d^2/k), num_class], bias is included
        if sample_rate == 0:
            self.sampler = None
        else:
            self.sampler = nn.Dropout(p=sample_rate)
        self.lossf = lossf

    def forward(self, head_embed: torch.Tensor = None, tail_embed: torch.Tensor = None, ht_embed: torch.Tensor = None,
                hs_type_embs: torch.Tensor = None, ts_type_embs: torch.Tensor = None):
        head_types = self.head_type_extra(hs_type_embs)
        tail_types = self.head_type_extra(ts_type_embs)
        head_embed = head_embed * head_types
        tail_embed = tail_embed * tail_types

        hs = torch.tanh(self.dropout(self.head_ext(torch.cat([head_embed, ht_embed], dim=-1))))
        ts = torch.tanh(self.dropout(self.tail_ext(torch.cat([tail_embed, ht_embed], dim=-1))))

        linear_logits = self.linear(torch.cat([hs, ts], dim=-1))
        hs = hs.view(-1, self.num_block, self.block_dim)
        ts = ts.view(-1, self.num_block, self.block_dim)
        bl = (hs.unsqueeze(3) * ts.unsqueeze(2)).view(-1, self.emb_dim * self.block_dim)
        bilinear_logits = self.bilinear(bl)
        logits = bilinear_logits + linear_logits
        # return bilinear_logits
        return logits

    def compute_loss(self, head_embed: torch.Tensor, tail_embed: torch.Tensor, ht_embed: torch.Tensor = None,
                     hs_type_embs: torch.Tensor = None, ts_type_embs: torch.Tensor = None,
                     labels: torch.Tensor = None, return_logit: bool = False):
        """
        all tensor with size [\sum{n^2}]
        """
        logits = self.forward(head_embed, tail_embed, ht_embed, hs_type_embs, ts_type_embs)
        if return_logit:
            return 0, logits
        sample_logits = logits
        sample_labels = labels
        # sample negative data
        if self.sampler is not None:
            if len(labels.size()) == 2:
                neg_mask = (labels[:, 1:].sum(dim=-1) == 0).float()
            else:
                neg_mask = (labels == 0).float()
            pos_mask = 1 - neg_mask
            mask = pos_mask + self.sampler(neg_mask)
            sample_logits = logits[mask == 1]
            sample_labels = labels[mask == 1]

        if isinstance(self.lossf, nn.BCEWithLogitsLoss):
            sample_labels = sample_labels.float()
            sample_logits = sample_logits.squeeze(-1)
        loss = self.lossf(sample_logits, sample_labels)
        return loss, logits
