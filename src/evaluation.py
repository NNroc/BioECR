# implementation of evaluation metrics
from src.metrics import muc, b_cubed, ceafe


def get_span_mapping(pred: list, gold: list):
    """
    Input in sample format. Get the mapping dict from pred to gold.
    Span format: (start, end)
    """
    span_mapping = {}
    tp = 0
    for i_p, ps in enumerate(pred):
        for i_g, gs in enumerate(gold):
            if ps == gs:
                span_mapping[i_p] = i_g
                tp += 1
                break
        if i_p not in span_mapping:
            span_mapping[i_p] = -1
    return tp, span_mapping


def count_similarity(pc, gc):
    # 计算两个列表中共同的元素数量
    common_elements = pc & gc
    # 计算gc的长度
    gc_length = len(gc)
    # 计算gc长度的一半
    half_gc_length = gc_length // 2 + 1
    # 检查共同元素的数量是否超过gc长度的一半
    if len(common_elements) >= half_gc_length and len(pc) <= len(gc):
        return 1
    else:
        return 0


def get_cluster_mapping(pred: list, gold: list):
    """
    Input in sample format. Get the mapping dict from pred to gold.
    Pred has already been converted by span_mapping.
    Cluster format: [...set of span index...]
    """
    cluster_mapping = {}
    tp = 0
    pred = [set(cluster) for cluster in pred]
    gold = [set(cluster) for cluster in gold]
    for i_p, pc in enumerate(pred):
        for i_g, gc in enumerate(gold):
            # if count_similarity(pc, gc):
            if pc == gc:
                cluster_mapping[i_p] = i_g
                tp += 1
                break
        if i_p not in cluster_mapping:
            cluster_mapping[i_p] = -1
    return tp, cluster_mapping


def get_cluster_mapping_real(pred: list, gold: list):
    """
    situations similar to those in real-life contexts.
    modify from get_cluster_mapping.
    """
    cluster_mapping = {}
    tp = 0
    pred = [set(cluster) for cluster in pred]
    gold = [set(cluster) for cluster in gold]
    has_use = set()
    for i_p, pc in enumerate(pred):
        for i_g, gc in enumerate(gold):
            # if count_similarity(pc, gc):
            if pc.issubset(gc):
                cluster_mapping[i_p] = i_g
                if i_g not in has_use:
                    has_use.add(i_g)
                    tp += 1
                break
        if i_p not in cluster_mapping:
            cluster_mapping[i_p] = pc
    return tp, cluster_mapping


def get_relation_mapping(pred: list, gold: list):
    """
    Input in sample format. Get the mapping dict from pred to gold.
    Pred has already been converted by cluster_mapping.
    Relation format: {'h': h, 'r': r, 't': t}
    """
    relation_mapping = {}
    tp = 0
    for i_p, pr in enumerate(pred):
        for i_g, gr in enumerate(gold):
            if pr == gr:
                relation_mapping[i_p] = i_g
                tp += 1
                break
        if i_p not in relation_mapping:
            relation_mapping[i_p] = -1
    return tp, relation_mapping


def compute_me_f1(pred: list, gold: list):
    """
    Input in batch format.
    """
    gold_cnt, pred_cnt, tp_cnt = 0, 0, 0
    for g, p in zip(gold, pred):
        g, p = list(set(g)), list(set(p))
        gold_cnt += len(g)
        pred_cnt += len(p)
        tp_cnt += get_span_mapping(p, g)[0]
    precision = tp_cnt / (pred_cnt + 1e-7)
    recall = tp_cnt / gold_cnt
    f1 = (precision * recall * 2) / (precision + recall + 1e-7)
    return f1, precision, recall


def compute_cr_f1(span_mappings=None, pred_clusters: list = None, gold_clusters: list = None, metric=None):
    """
    Input in batch format. Including span_mappings.
    """
    gold_cnt, pred_cnt, tp_cnt = 0, 0, 0
    if span_mappings is None:  # pred_spans == gold_spans
        if metric is None:
            for p_clusters, g_clusters in zip(pred_clusters, gold_clusters):
                gold_cnt += len(g_clusters)
                pred_cnt += len(p_clusters)
                tp_cnt += get_cluster_mapping(p_clusters, g_clusters)[0]
            precision = tp_cnt / (pred_cnt + 1e-7)
            recall = tp_cnt / gold_cnt
        elif metric in [muc, b_cubed]:
            p_num, p_den, r_num, r_den = 0, 0, 0, 0
            for p_clusters, g_clusters in zip(pred_clusters, gold_clusters):
                pn, pd = metric(p_clusters, g_clusters)
                rn, rd = metric(g_clusters, p_clusters)
                p_num += pn
                p_den += pd
                r_num += rn
                r_den += rd
            precision = p_num / (p_den + 1e-7)
            recall = r_num / (r_den + 1e-7)
        elif metric == ceafe:
            p_num, p_den, r_num, r_den = 0, 0, 0, 0
            for p_clusters, g_clusters in zip(pred_clusters, gold_clusters):
                pn, pd, rn, rd = metric(p_clusters, g_clusters)
                p_num += pn
                p_den += pd
                r_num += rn
                r_den += rd
            pn, pd, rn, rd = metric(pred_clusters, gold_clusters)
            precision = pn / (pd + 1e-7)
            recall = rn / (rd + 1e-7)
        else:
            raise ValueError("Unknown CR metric.")
    else:
        for span_mapping, p_clusters, g_clusters in zip(span_mappings, pred_clusters, gold_clusters):
            # first find a mapping between pred cluster and gold cluster
            new_p_clusters = []
            for pc in p_clusters:
                new_p_clusters.append([span_mapping[i_s] for i_s in pc])
            # calculation
            gold_cnt += len(g_clusters)
            pred_cnt += len(p_clusters)
            tp_cnt += get_cluster_mapping(new_p_clusters, g_clusters)[0]
        precision = tp_cnt / (pred_cnt + 1e-7)
        recall = tp_cnt / gold_cnt
    f1 = (precision * recall * 2) / (precision + recall + 1e-7)
    return f1, precision, recall


def compute_avg_cr_f1(span_mappings=None, pred_clusters: list = None, gold_clusters: list = None):
    if span_mappings is not None:
        return compute_cr_f1(span_mappings, pred_clusters, gold_clusters)
    muc_f1, _, _ = compute_cr_f1(None, pred_clusters, gold_clusters, muc)
    b3_f1, _, _ = compute_cr_f1(None, pred_clusters, gold_clusters, b_cubed)
    ceafe_f1, _, _ = compute_cr_f1(None, pred_clusters, gold_clusters, ceafe)
    avg_f1 = sum([muc_f1, b3_f1, ceafe_f1]) / 3
    return avg_f1, muc_f1, b3_f1, ceafe_f1


def compute_re_f1_pubtator(cluster_mappings=None, pred_relations: list = None, gold_relations: list = None,
                           vertexSets: list = None):
    gold_cnt, pred_cnt, tp_cnt, ign_cnt = 0, 0, 0, 0
    tp_list = []
    for cluster_mapping, p_relations, g_relations, vertexSet in zip(cluster_mappings, pred_relations,
                                                                    gold_relations, vertexSets):
        new_p_relations = [{'h': cluster_mapping[r['h']], 't': cluster_mapping[r['t']], 'r': r['r']} for r in
                           p_relations]
        gold_cnt += len(g_relations)
        pred_cnt += len(p_relations)
        tp_li = []
        for pr in new_p_relations:
            prr = {'h': pr['t'], 't': pr['h'], 'r': pr['r']}
            if pr in g_relations or prr in g_relations:
                tp_cnt += 1
                tp_li.append(1)
            else:
                tp_li.append(0)
        tp_list.append(tp_li)
    precision = tp_cnt / (pred_cnt + 1e-7)
    recall = tp_cnt / (gold_cnt + 1e-7)
    f1 = (precision * recall * 2) / (precision + recall + 1e-7)
    return f1, precision, recall, tp_cnt, pred_cnt, gold_cnt, tp_list


def compute_re_f1_pubtator_real(cluster_mappings=None, pred_relations: list = None, gold_relations: list = None,
                                vertexSets: list = None):
    gold_cnt, pred_cnt, tp_cnt, ign_cnt, tp_list = 0, 0, 0, 0, []
    for cluster_mapping, p_relations, g_relations, vertexSet in zip(cluster_mappings, pred_relations,
                                                                    gold_relations, vertexSets):
        new_p_re = []
        new_p_re_set = set()
        for r in p_relations:
            new_p_re.append({'h': cluster_mapping[r['h']], 't': cluster_mapping[r['t']], 'r': r['r']})
            new_p_re_set.add(str({'h': cluster_mapping[r['h']], 't': cluster_mapping[r['t']], 'r': r['r']}))
        gold_cnt += len(g_relations)
        pred_cnt += len(new_p_re_set)
        tp_li = []
        tp_set = set()
        for pr in new_p_re:
            prr = {'h': pr['t'], 't': pr['h'], 'r': pr['r']}
            if pr in g_relations or prr in g_relations:
                tp_li.append(1)
                if str(pr) not in tp_set and str(prr) not in tp_set:
                    tp_set.add(str(pr))
                    tp_cnt += 1
            else:
                tp_li.append(0)
        tp_list.append(tp_li)
    precision = tp_cnt / (pred_cnt + 1e-7)
    recall = tp_cnt / (gold_cnt + 1e-7)
    f1 = (precision * recall * 2) / (precision + recall + 1e-7)
    return f1, precision, recall, tp_cnt, tp_list
