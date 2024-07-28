import argparse
import json
import os


def generate_gc_data(dev_file, test_file, dataset='docred'):
    """
    Generate data for GC (graph composition, i.e. coreference resolution + relation extraction).
    Use span prediction from ME.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, f'data/{dataset}/')
    # process train
    raw_train = json.load(open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['vertexSet']:
            entity_len.append(len(e))
            for m in e:
                ms = [[m['sent_id'], m['pos'][0]], [m['sent_id'], m['pos'][1]], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, 'train_gc.json'), 'w', encoding='utf-8'))
    # process dev/test
    for split, spanfile in zip(['dev', 'test'], [dev_file, test_file]):
        raw_data = json.load(open(os.path.join(data_dir, '{}.json'.format(split)), 'r', encoding='utf-8'))
        span_data = []
        with open(spanfile, 'r') as f:
            for line in f:
                span_data.append(json.loads(line))
        assert len(span_data) == len(raw_data)
        for i, sample in enumerate(raw_data):
            spans = span_data[i]['tp'] + span_data[i]['fp']
            sample['spans'] = spans
        json.dump(raw_data, open(os.path.join(data_dir, '{}_gc.json'.format(split)), 'w', encoding='utf-8'))


def generate_gc_data_pubtator(dev_file, test_file, dataset='BioRED'):
    """
    Generate data for GC (graph composition, i.e. coreference resolution + relation extraction).
    Use span prediction from ME.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, f'data/{dataset}/pre')
    file_name = 'train'
    # process train
    raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['entities']:
            entity_len.append(len(e))
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))
    # process dev/test
    for split, spanfile in zip(['dev', 'test'], [dev_file, test_file]):
        raw_data = json.load(open(os.path.join(data_dir, '{}.json'.format(split)), 'r', encoding='utf-8'))
        span_data = []
        with open(spanfile, 'r') as f:
            for line in f:
                span_data.append(json.loads(line))
        assert len(span_data) == len(raw_data)
        for i, sample in enumerate(raw_data):
            spans = span_data[i]['tp'] + span_data[i]['fp']
            sample['spans'] = spans
        json.dump(raw_data, open(os.path.join(data_dir, '{}_gc.json'.format(split)), 'w', encoding='utf-8'))


def generate_gc_data_pubtator_no_dev(test_file, dataset='BioRED'):
    """
    Generate data for GC (graph composition, i.e. coreference resolution + relation extraction).
    Use span prediction from ME.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, f'data/{dataset}/pre')

    # process train
    file_name = 'train'
    raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['entities']:
            entity_len.append(len(e))
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))

    # process dev
    file_name = 'dev'
    raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['entities']:
            entity_len.append(len(e))
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))

    # # process test
    # file_name = 'test'
    # raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    # for sample in raw_train:
    #     entity_len = []
    #     spans = []
    #     for e in sample['entities']:
    #         entity_len.append(len(e))
    #         for m in e:
    #             sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
    #             sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
    #             ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
    #             spans.append(ms)
    #     sample['entity_len'] = entity_len
    #     sample['spans'] = spans
    # json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))

    # process test
    for split, spanfile in zip(['test'], [test_file]):
        raw_data = json.load(open(os.path.join(data_dir, '{}.json'.format(split)), 'r', encoding='utf-8'))
        span_data = []
        with open(spanfile, 'r') as f:
            for line in f:
                span_data.append(json.loads(line))
        assert len(span_data) == len(raw_data)
        for i, sample in enumerate(raw_data):
            spans = span_data[i]['tp'] + span_data[i]['fp']
            sample['spans'] = spans
        json.dump(raw_data, open(os.path.join(data_dir, '{}_gc.json'.format(split)), 'w', encoding='utf-8'))


def generate_gc_data_pubtator_gold(dataset):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, f'data/{dataset}/pre')

    # process train
    file_name = 'train'
    raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['entities']:
            entity_len.append(len(e))
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))

    # process dev
    file_name = 'dev'
    raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['entities']:
            entity_len.append(len(e))
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))

    # process test
    file_name = 'test'
    raw_train = json.load(open(os.path.join(data_dir, file_name + '.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['entities']:
            entity_len.append(len(e))
            for m in e:
                sent_id_s, sent_pos_id_s = pos2sents_id(sample['sents'], m['pos'][0])
                sent_id_e, sent_pos_id_e = pos2sents_id(sample['sents'], m['pos'][1])
                ms = [[sent_id_s, sent_pos_id_s], [sent_id_e, sent_pos_id_e], m['type']]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, file_name + '_gc.json'), 'w', encoding='utf-8'))


def delete_invalid_spans(src, dst):
    data = []
    tp, fp, fn = 0, 0, 0
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    for entry in data:
        new_spans = []
        tp += len(entry['tp'])
        fn += len(entry['fn'])
        for span in entry['fp']:
            if order_score(span[0]) >= order_score(span[1]):
                print([span[0], span[1]])
                continue
            new_spans.append(span)
            fp += 1
        entry['fp'] = new_spans
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = p * r * 2 / (p + r + 1e-7)
    print('tp', tp, 'fp', fp, 'fn', fn)
    print('p', p, 'r', r, 'f1', f1)


def order_score(pos):
    return pos[0] * 100 + pos[1]


def pos2sents_id(sents, pos_id):
    sent_id = 0
    sent_pos_id = pos_id
    for sent in sents:
        if sent_pos_id >= len(sent):
            sent_id += 1
            sent_pos_id -= len(sent)
        else:
            break
    return sent_id, sent_pos_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GDA")
    parser.add_argument("--no_dev", type=int, default=0)
    parser.add_argument("--perfect", type=int, default=0)
    args = parser.parse_args()
    dev_file, test_file = None, None
    if args.perfect:
        # 完美数据集
        generate_gc_data_pubtator_gold(args.dataset)
        print('generate perfect dataset')
    elif args.dataset == 'CDR' and args.no_dev:
        # 完整的
        test_file = "./result/CDR/me/CDR-me_cr_re-me_test.json"
        generate_gc_data_pubtator_no_dev(test_file, args.dataset)
        delete_invalid_spans(test_file, test_file)
    elif args.dataset == 'BioRED' and args.no_dev:
        # 完整的
        test_file = "./result/BioRED/me/BioRED-me_cr_re-me_test.json"
        generate_gc_data_pubtator_no_dev(test_file, args.dataset)
        delete_invalid_spans(test_file, test_file)
    elif args.dataset == 'GDA':
        # 完整的
        dev_file = "./result/GDA/me/GDA-me_cr_re-me_dev.json"
        test_file = "./result/GDA/me/GDA-me_cr_re-me_test.json"
        generate_gc_data_pubtator(dev_file, test_file, args.dataset)
        delete_invalid_spans(dev_file, dev_file)
        delete_invalid_spans(test_file, test_file)
    else:
        print('error')
