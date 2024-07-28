import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from .base import BaseModel


class CoreferenceResolutionModel(BaseModel):
    def __init__(self, hidden_dim: int, block_dim: int = 128, threshold: float = 0.4, strategy: str = 'average'):
        # threshold: the minimum accepted similarity
        super().__init__(hidden_dim, hidden_dim, block_dim, 1, .0, nn.BCEWithLogitsLoss())
        self.clustering = AgglomerativeClustering(n_clusters=None, linkage=strategy, affinity='precomputed',
                                                  distance_threshold=1 - threshold)

    def inference(self, head_embed: torch.Tensor = None, tail_embed: torch.Tensor = None,
                  span_len=None, batch_hts=None, logits: torch.Tensor = None, type_input_list=None):
        offset = 0
        predictions = []
        if logits is None:
            logits = self.forward(head_embed, tail_embed).to(dtype=torch.float64)
            logits = torch.sigmoid(logits)  # convert logits to probabilities
        for l, hts, type_input in zip(span_len, batch_hts, type_input_list):
            assert l * l == len(hts), "span_len^2 must equal to len(hts)."
            # one entity
            if l == 1:
                predictions.append([[0], type_input[0]])
                continue
            preds = logits[offset:offset + l * l]
            preds = preds.squeeze(-1)
            offset += l * l

            indices = [[ht_i[0] for ht_i in hts], [ht_i[1] for ht_i in hts]]
            indices = [torch.tensor(x).to(preds.device) for x in indices]
            table = torch.zeros(l, l).to(preds)
            table.index_put_(indices, preds)
            predictions.append(self.__decode_single_table(table, type_input))
        return predictions

    def __decode_single_table(self, similarity: torch.Tensor, type_input: list):
        # table: n*n size
        # step 1: average from diagnal
        similarity = (similarity + similarity.permute(1, 0)) / 2
        similarity = similarity.cpu().numpy()
        distance = 1 - similarity
        for i in range(len(distance)):
            for j in range(len(distance)):
                if type_input[i] != type_input[j]:
                    distance[i][j] = distance[i][j] + 1
        assignment = self.clustering.fit_predict(distance)
        clusters = [[] for i in range(assignment.max() + 1)]
        types_map = [{} for i in range(assignment.max() + 1)]
        types = [[] for i in range(assignment.max() + 1)]
        for i_s, i_c in enumerate(assignment):
            clusters[i_c].append(i_s)
            if type_input[i_s] in types_map[i_c]:
                types_map[i_c][type_input[i_s]] += 1
            else:
                types_map[i_c][type_input[i_s]] = 1
        for i_s, type_map in enumerate(types_map):
            max_key = max(type_map, key=type_map.get)
            types[i_s] = max_key
        return clusters, types
