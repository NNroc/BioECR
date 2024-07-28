import json
import torch
import os
from src.prepro import *
from tqdm import tqdm
from itertools import accumulate
from src.modules.config import *

# find project abs dir
root_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(root_dir)
entity_types_description_path = os.path.join(root_dir, "data/entity_types.jsonl")


def get_type_input_ids(tokenizer, dataset):
    type_input_ids = []
    with open(entity_types_description_path, 'r', encoding="utf-8") as f:
        for line in f:
            description = json.loads(line)
            if description["dataset"] == dataset:
                type_input = description["description"]
                type_input = tokenizer.tokenize(type_input)
                type_input = tokenizer.convert_tokens_to_ids(type_input)
                type_input = tokenizer.build_inputs_with_special_tokens(type_input)
                type_input_ids.append(type_input)
    return type_input_ids


def read_dataset(tokenizer, filename='train.json', dataset='CDR', task='me', no_dev=0):
    if task == 'me':
        return read_me_data_pubtator(tokenizer=tokenizer, filename=filename, dataset=dataset)
    elif task == 'gc':
        return read_gc_data_pubtator(tokenizer=tokenizer, filename=filename, dataset=dataset, no_dev=no_dev)
    raise ValueError("Unknown task type.")


def read_me_data_pubtator(tokenizer, filename='train.json', dataset='CDR'):
    if dataset == 'BioRED':
        e_types_description_name2id = e_types_description_name2id_biored
    elif dataset == 'CDR':
        e_types_description_name2id = e_types_description_name2id_cdr
    elif dataset == 'GDA':
        e_types_description_name2id = e_types_description_name2id_gda
    else:
        raise ValueError("dataset name error")
    type_input_ids = get_type_input_ids(tokenizer, dataset)
    if len(type_input_ids) < 1:
        raise ValueError("Unknown dataset.")
    e_type_num = len(e_types_description_name2id)
    tag = tag_default
    i_line = 0
    features = []
    bias_mentions, bias_entities = 0, 0  # count the mention that been discard
    cnt_mention = 0
    data_path = 'data/{}/pre/{}'.format(dataset, filename)
    data = json.load(open(os.path.join(root_dir, data_path), 'r', encoding='utf-8'))

    offset = 1  # for BERT/RoBERTa/Longformer, all follows [CLS] sent [SEP] format
    for sample in tqdm(data, desc='Data'):
        i_line += 1
        sents = []
        doc_map = []
        doc_map_1list = {}
        entities = sample['entities']
        spans = []
        doc_map_1list_id = 0
        for i_s, sent in enumerate(sample['sents']):
            sent_map = {}  # map the old index to the new index
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                sent_map[i_t] = len(sents)
                doc_map_1list[doc_map_1list_id] = len(sents)
                doc_map_1list_id += 1
                sents.extend(tokens_wordpiece)
            sent_map[len(sent)] = len(sents)
            doc_map.append(sent_map)

        label = [[tag["O"] for _ in range(len(sents))] for _ in range(e_type_num)]
        entity_len = []
        entity_type2id = {}
        for e in entities:
            entity_type2id[e[0]['type_id'][0]] = len(entity_type2id)
            flag_entity = False
            e_new = set()  # first: remove duplicate mentions for single entity
            for m in e:
                e_new.add((m["sent_id"], m["pos"][0], m["pos"][1], m['type']))
            entity_len.append(len(e_new))  # save entity lens to later construct cr_label
            cnt_mention += len(e_new)
            for m in e_new:
                flag_mention = False
                start, end = doc_map_1list[m[1]], doc_map_1list[m[2]]
                # add this span to mr_spans
                if (start + offset, end + offset, m[3]) in spans:
                    continue
                spans.append((start + offset, end + offset, m[3]))
                # e_type_id
                m_type_id = e_types_description_name2id[m[3]]
                for j in range(start, end):
                    if label[m_type_id][j] != 0:
                        bias_mentions += 1
                        flag_mention = True
                        break
                if flag_mention:
                    flag_entity = True
                    continue
                # label[m_type_id][start] = tag["B-" + m[3]]
                label[m_type_id][start] = tag["B"]
                for j in range(start + 1, end):
                    # label[m_type_id][j] = tag["I-" + m[3]]
                    label[m_type_id][j] = tag["I"]
            if flag_entity:
                bias_entities += 1

        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        label = [[tag["O"]] + label[_] + [tag["O"]] for _ in range(e_type_num)]
        feature = {"input_ids": input_ids, "type_input_ids": type_input_ids, "label": label, "spans": spans,
                   "doc_map": doc_map, "doc_len": len(input_ids)}
        features.append(feature)

    print("# of documents:\t\t{}".format(i_line))
    print("# of mention bias:\t{}".format(bias_mentions))
    print("# of entity bias:\t{}".format(bias_entities))
    print("# of mentions:\t\t{}".format(cnt_mention))
    return features


def read_gc_data_pubtator(tokenizer, filename='train.json', dataset='BioRED', no_dev=0):
    max_input_size, max_entities = 0, 0
    i_line = 0
    features = []
    data_path = f'data/{dataset}/pre/{filename}'
    data = json.load(open(os.path.join(root_dir, data_path), 'r', encoding='utf-8'))
    type_input_ids = get_type_input_ids(tokenizer, dataset)
    if len(type_input_ids) < 1:
        raise ValueError("Unknown dataset.")
    if dataset == 'BioRED':
        rel2id = rel2id_biored
        pairs2rel = pairs2rel_biored
        e_types_description_name2id = e_types_description_name2id_biored
    elif dataset == 'CDR':
        rel2id = rel2id_cdr
        pairs2rel = pairs2rel_cdr
        e_types_description_name2id = e_types_description_name2id_cdr
    elif dataset == 'GDA':
        rel2id = rel2id_gda
        pairs2rel = pairs2rel_gda
        e_types_description_name2id = e_types_description_name2id_gda
    else:
        raise ValueError("Unknown dataset.")
    offset = 1  # for BERT/RoBERTa/Longformer, all follows [CLS] sent [SEP] format
    for sample in tqdm(data, desc='Data'):
        i_line += 1
        sents = []
        doc_map = []
        span_start, span_end, span_type, type_input_list = [], [], [], []
        raw_spans_2 = [((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in sample['spans']]
        raw_spans_3 = [((s[0][0], s[0][1]), (s[1][0], s[1][1]), s[2]) for s in sample['spans']]
        spans2entities = []
        for s in raw_spans_3:
            span_start.append(s[0])
            span_end.append(s[1])
            span_type.append(s[2])
            type_input_list.append(e_types_description_name2id[s[2]])
        # index : type_id
        entity_typeid2index, i_index = {}, 0
        for i_entities in sample['entities']:
            entity_typeid2index[i_index] = i_entities[0]['type_id']
            i_index += 1
        entity_index2typeid = {value: key for key, value in entity_typeid2index.items()}
        # get doc_map, input_ids
        for i_s, sent in enumerate(sample['sents']):
            sent_map = {}  # map the old index to the new index
            for i_t, token in enumerate(sent):
                # lower
                token = token.lower()
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in span_start:
                    span_index = span_start.index((i_s, i_t))
                    re_type = rel_type2rel_type_description[span_type[span_index]]
                    tokens_wordpiece = ["<<" + re_type + ">>"] + tokens_wordpiece
                if (i_s, i_t + 1) in span_end:
                    span_index = span_end.index((i_s, i_t + 1))
                    re_type = rel_type2rel_type_description[span_type[span_index]]
                    tokens_wordpiece = tokens_wordpiece + ["<<" + re_type + ">>"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            sent_map[i_t + 1] = len(sents)
            doc_map.append(sent_map)
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        # get spans
        spans = []
        for rs in raw_spans_3:
            start = doc_map[rs[0][0]][rs[0][1]]
            end = doc_map[rs[1][0]][rs[1][1]]
            spans.append((start + offset, end + offset))
            spaninsent = sample['sents'][rs[0][0]]
            spans2entities.append(([spaninsent[en] for en in range(rs[0][1], rs[1][1])], rs[2]))
        # 各种图及掩码
        graphs = {}
        # get clusters and relations
        # if mention not in predictions, add -1 as index
        clusters, relations = [], []
        cr_label, re_label, re_table_label = None, None, None
        hts = None
        if 'labels' in sample:
            for hrt in sample['labels']:
                h = entity_index2typeid[hrt['h']]
                t = entity_index2typeid[hrt['t']]
                r = rel2id[hrt['r']]
                relations.append({'h': h, 'r': r, 't': t})
        entities = sample['entities']
        for e in entities:
            cluster = []
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms_2 = ((sent_id_s, sent_pos_id_s), (sent_id_e, sent_pos_id_e))
                ms_3 = ((sent_id_s, sent_pos_id_s), (sent_id_e, sent_pos_id_e), m['type'])
                # ms = ((m['sent_id'], m['pos'][0]), (m['sent_id'], m['pos'][1]))
                if ms_3 not in raw_spans_3:
                    cluster.append(-1)
                else:
                    cluster.append(raw_spans_3.index(ms_3))
            clusters.append(cluster)

        # get graph
        # get syntax graph: entities in same sent
        syntax_graph = torch.zeros(len(spans), len(spans))
        for i in range(len(spans)):
            for j in range(len(spans)):
                if raw_spans_3[i][0][0] == raw_spans_3[j][0][0]:
                    syntax_graph[i][j] = 1
        # syntax_graph[syntax_graph == 0] = -1e30
        # syntax_graph = torch.softmax(syntax_graph, dim=-1)
        syntax_graph = syntax_graph.cpu().tolist()
        graphs['syntax_graph'] = syntax_graph
        # get mentions type graph: entities with same type
        mentions_type_graph = torch.zeros(len(spans), len(spans))
        for i in range(len(spans)):
            for j in range(len(spans)):
                if raw_spans_3[i][2] == raw_spans_3[j][2]:
                    mentions_type_graph[i][j] = 1
        mentions_type_graph = mentions_type_graph.cpu().tolist()
        graphs['mentions_type_graph'] = mentions_type_graph

        if 'train' in filename or (no_dev and 'dev' in filename):
            # construct ht_to_r dict
            ht_to_r = dict()
            for hrt in relations:
                h, r, t = hrt['h'], hrt['r'], hrt['t']
                if (h, t) in ht_to_r:
                    ht_to_r[(h, t)].append(r)
                else:
                    ht_to_r[(h, t)] = [r]
            # construct label
            entity_len = sample['entity_len']
            accumulate_entity_len = [0] + [i for i in accumulate(entity_len)]
            hts, cr_label, re_label, re_table_label = [], [], [], []
            for h_e in range(len(entities)):
                for t_e in range(len(entities)):
                    re_table_flag = 0
                    if (h_e, t_e) in ht_to_r or (t_e, h_e) in ht_to_r:
                        re_table_flag = 1
                    if h_e == t_e:
                        cr_flag = 1
                        relation = [1] + [0] * (len(rel2id) - 1)
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                hts.append([h_m, t_m])
                                cr_label.append(cr_flag)
                                re_label.append(relation)
                                re_table_label.append(re_table_flag)
                        continue
                    # add negative cr_label
                    cr_flag = 0
                    if (h_e, t_e) in ht_to_r:
                        relation = [0] * len(rel2id)
                        for r in ht_to_r[(h_e, t_e)]:
                            relation[r] = 1
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                hts.append([h_m, t_m])
                                cr_label.append(cr_flag)
                                re_label.append(relation)
                                re_table_label.append(re_table_flag)
                    else:
                        relation = [1] + [0] * (len(rel2id) - 1)
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                hts.append([h_m, t_m])
                                cr_label.append(cr_flag)
                                re_label.append(relation)
                                re_table_label.append(re_table_flag)

            tmp = list(zip(hts, re_label, cr_label))
            tmp = sorted(tmp, key=lambda x: 100 * x[0][0] + x[0][1])
            hts, re_label, cr_label = zip(*tmp)
            hts, re_label, cr_label = list(hts), list(re_label), list(cr_label)
            # make sure all data are in table format, i.e. can be view() to transfer to table
            assert len(hts) == len(re_label) == len(cr_label) == len(spans) * len(spans) \
                   == len(type_input_list) * len(type_input_list), "{} must match {} must match {}.".format(
                len(hts), len(spans) * len(spans), len(type_input_list) * len(type_input_list))
        feature = {"input_ids": input_ids, "spans": spans, "hts": hts, "spans2entities": spans2entities,
                   "type_input_ids": type_input_ids, "type_input_list": type_input_list,
                   "cr_label": cr_label, "cr_clusters": clusters,
                   "re_label": re_label, "re_triples": relations, "re_table_label": re_table_label,
                   "graphs": graphs, "vertexSet": entities, "title": sample["title"]}
        features.append(feature)
    print("# of documents:\t\t{}.".format(i_line))
    return features
