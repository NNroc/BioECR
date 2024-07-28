import torch
import torch.nn as nn

from src.modules.base import BaseModel, get_hrt, gen_hts, convert_table_to_node, convert_graph_to_table
from src.modules.mention_extraction import MeModel
from src.modules.coreference_resolution import CoreferenceResolutionModel
from src.modules.relation_extraction import RelationModel
from src.modules.graph import RelGCN, CompGCN
from src.long_seq import process_long_input, process_multiple_segments


class MEModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.bert = encoder
        self.hidden_size = config.hidden_size
        self.me_model = MeModel(self.hidden_size, 0.3, config.dataset)

    def encode(self, input_ids, attention_mask):
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output

    def compute_loss(self, input_ids=None, attention_mask=None, type_input_ids=None, type_attention_mask=None,
                     label=None):
        sequence_output = self.encode(input_ids, attention_mask)
        type_sequence_output = self.encode(type_input_ids, type_attention_mask)[:, 0]
        loss1 = self.me_model.compute_loss(sequence_output, attention_mask,
                                           type_sequence_output, type_attention_mask, label)
        return loss1

    def inference(self, input_ids=None, attention_mask=None, type_input_ids=None, type_attention_mask=None):
        sequence_output = self.encode(input_ids, attention_mask)
        type_sequence_output = self.encode(type_input_ids, type_attention_mask)[:, 0]
        preds = self.me_model.inference(sequence_output, attention_mask, type_sequence_output, type_attention_mask)
        return preds


class CREModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.bert = encoder
        self.hidden_size = config.hidden_size
        self.CRTablePredictor = BaseModel(hidden_dim=self.hidden_size, emb_dim=self.hidden_size,
                                          dataset=config.dataset, dropout=config.dropout,
                                          block_dim=64, num_class=1, sample_rate=0, lossf=nn.BCEWithLogitsLoss())
        self.RETablePredictor = BaseModel(hidden_dim=self.hidden_size, emb_dim=self.hidden_size,
                                          dataset=config.dataset, dropout=config.dropout,
                                          block_dim=64, num_class=1, sample_rate=0, lossf=nn.BCEWithLogitsLoss())
        self.RGCN = RelGCN(self.hidden_size, self.hidden_size, self.hidden_size, num_rel=3,
                           num_layer=config.num_gcn_layers)
        self.CompGCN = CompGCN(self.hidden_size, self.hidden_size, dropout=config.dropout, num_rels=4,
                               conv_num=config.conv_num, opn=config.opn)
        self.CR = CoreferenceResolutionModel(self.hidden_size)
        self.RE = RelationModel(self.hidden_size, config.num_class, beta=config.beta, max_pred=-1,
                                dataset=config.dataset)
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.adj_lambda = config.adj_lambda

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output, attention

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def forward(self, input_ids=None, attention_mask=None, type_input_ids=None, type_attention_mask=None,
                type_input_list=None, spans=None, graphs=None, hts=None,
                cr_label=None, re_label=None, cr_table_label=None, re_table_label=None, return_logit=False):
        # 4 12 414 414
        sequence_output, attention = self.encode(input_ids, attention_mask)
        type_sequence_output, type_attention = self.encode(type_input_ids, type_attention_mask)
        type_sequence_output = type_sequence_output[:, 0]
        # type_sequence_output = []
        span_len = [len(span) for span in spans]
        sg_adj = graphs["syntax_graph"].to(sequence_output.device)
        type_adj = graphs["mentions_type_graph"].to(sequence_output.device)
        if hts is None:
            hts = [gen_hts(l) for l in span_len]

        hs, ts, rs, span_embs, span_atts, hs_type_embs, ts_type_embs = get_hrt(sequence_output, attention, spans, hts,
                                                                               type_sequence_output, type_input_list)
        cr_table_loss, cr_table = self.CRTablePredictor.compute_loss(hs, ts, rs, hs_type_embs, ts_type_embs,
                                                                     labels=cr_table_label, return_logit=return_logit)
        re_table_loss, re_table = self.RETablePredictor.compute_loss(hs, ts, rs, hs_type_embs, ts_type_embs,
                                                                     labels=re_table_label, return_logit=return_logit)

        # convert logits to tabel in batch form
        cr_adj, re_adj = convert_table_to_node(cr_table, re_table, span_len)
        adjacency_list = [cr_adj, re_adj, sg_adj, type_adj]
        nodes = span_embs
        # compgcn
        adjacency_list = [convert_graph_to_table(adj, span_len) for adj in adjacency_list]
        nodes_h_t = self.CompGCN(nodes, hs, ts, rs, adjacency_list, self.adj_lambda)
        chunks = torch.chunk(nodes_h_t, 2, dim=0)
        hs, ts = chunks[0], chunks[1]

        cr_loss, cr_logits = self.CR.compute_loss(hs, ts, rs, hs_type_embs, ts_type_embs,
                                                  labels=cr_label, return_logit=return_logit)
        re_loss, re_logits = self.RE.compute_loss(hs, ts, rs, hs_type_embs, ts_type_embs,
                                                  labels=re_label, return_logit=return_logit)

        if return_logit:
            return cr_logits, re_logits
        else:
            return cr_loss, re_loss, cr_table_loss, re_table_loss

    def compute_loss(self, input_ids=None, attention_mask=None, type_input_ids=None, type_attention_mask=None,
                     type_input_list=None, spans=None, hts=None, cr_label=None, re_label=None,
                     cr_table_label=None, re_table_label=None, graphs=None):
        cr_loss, re_loss, cr_table_loss, re_table_loss = \
            self.forward(input_ids=input_ids, attention_mask=attention_mask,
                         type_input_ids=type_input_ids, type_input_list=type_input_list,
                         type_attention_mask=type_attention_mask, spans=spans, hts=hts, cr_label=cr_label,
                         re_label=re_label, cr_table_label=cr_table_label, re_table_label=re_table_label,
                         graphs=graphs)
        return cr_loss + re_loss + self.alpha * cr_table_loss + self.alpha * re_table_loss

    def inference(self, input_ids=None, attention_mask=None, type_input_ids=None, type_attention_mask=None,
                  type_input_list=None, spans=None, graphs=None):
        span_len = [len(span) for span in spans]
        hts = [gen_hts(l) for l in span_len]

        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "type_input_ids": type_input_ids,
                  "type_attention_mask": type_attention_mask,
                  "type_input_list": type_input_list,
                  "spans": spans,
                  "graphs": graphs}

        cr_logits, re_logits = self.forward(**inputs, return_logit=True)
        cr_logits = cr_logits.to(dtype=torch.float64)
        cr_logits = torch.sigmoid(cr_logits)
        re_logits = re_logits.to(dtype=torch.float64)
        cr_predictions = self.CR.inference(span_len=span_len, batch_hts=hts, logits=cr_logits,
                                           type_input_list=type_input_list)
        re_predictions = self.RE.inference(span_len=span_len, batch_hts=hts, batch_clusters=cr_predictions,
                                           logits=re_logits)
        outputs = {'cr_predictions': cr_predictions, 're_predictions': re_predictions}
        return outputs
