import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *


class MeModel(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.3, dataset: dict = None):
        """
        follow the BIO setting.
        """
        super(MeModel, self).__init__()
        if dataset == 'BioRED':
            self.id2e_types = id2e_types_description_name_biored
        elif dataset == 'CDR':
            self.id2e_types = id2e_types_description_name_cdr
        elif dataset == 'GDA':
            self.id2e_types = id2e_types_description_name_gda

        self.tag_dataset = tag_default
        self.tag_dataset = {value: key for key, value in self.tag_dataset.items()}
        self.hidden_dim = hidden_dim
        self.classifier = nn.Linear(hidden_dim, len(self.tag_dataset))
        self.activate_func = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.lossf = nn.CrossEntropyLoss()

    # input size: [batch_size, doc_len, hidden_dim]
    def forward(self, input: torch.Tensor, type_input: torch.Tensor):
        input_2 = self.dropout(self.activate_func(input))
        type_input_2 = self.activate_func(type_input)
        logits1 = type_input_2.unsqueeze(0).unsqueeze(2) * input_2.unsqueeze(1)
        logits = self.classifier(logits1)
        logits = logits.transpose(1, 2)
        return logits

    def compute_loss(self, input: torch.Tensor, attention_mask: torch.Tensor,
                     type_input: torch.Tensor, type_attention_mask: torch.Tensor, labels: torch.Tensor):
        # labels here: sequence tensor
        logits = self.forward(input, type_input)  # [batch_size, doc_len, num_class]
        # delete padding & resize
        logits = logits[attention_mask == 1]
        labels = labels.transpose(1, 2)
        labels = labels[attention_mask == 1]
        logits = logits.view(-1, len(self.tag_dataset))
        labels = labels.view(-1, )
        # compute cross entropy loss
        loss = self.lossf(logits, labels)
        return loss

    def inference(self, input, attention_mask, type_input: torch.Tensor, type_attention_mask: torch.Tensor):
        """
        return a list of list of tuple, containing predicted span results.

        @return: [[(start_i, end_i), ...], ...]
        """
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        pred_logit = self.forward(input, type_input)
        pred_class = torch.max(pred_logit, dim=-1)[1]
        pred_class = pred_class.cpu().numpy().astype(np.int32)
        pred_spans = [[] for i in range(len(seq_len))]
        for i, l_i in enumerate(seq_len):
            for type_id in range(pred_class.shape[2]):
                flag = False  # whether in a mention or not
                start = -1  # start position of mention
                label = ''
                for j in range(1, l_i):
                    if pred_class[i][j][type_id] == 0:  # O case
                        if flag:
                            pred_spans[i].append((start, j, label))
                            flag = False
                    elif pred_class[i][j][type_id] % 2 == 1:  # B case
                        if flag:
                            pred_spans[i].append((start, j, label))
                        else:
                            flag = True
                            # label = self.tag_dataset[pred_class[i][j][type_id]].split('-')[1]
                            label = self.id2e_types[type_id]
                        start = j
                    elif pred_class[i][j][type_id] % 2 == 0:  # I case
                        pass
                    else:
                        raise ValueError("Unexpected predictions in ME.")
        return pred_spans
