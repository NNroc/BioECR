import os
import re
from tqdm import tqdm
from recordtype import recordtype
from collections import OrderedDict
import argparse
import pickle
import json
from itertools import permutations, combinations
from tools import sentence_split_genia, tokenize_genia, adjust_offsets, find_mentions, find_cross, fix_sent_break, \
    generate_pairs, generate_pairs_multi_entities
from readers import *

TextStruct = recordtype('TextStruct', 'pmid txt')
EntStruct = recordtype('EntStruct', 'pmid name off1 off2 type kb_id sent_no word_id bio')
RelStruct = recordtype('RelStruct', 'pmid type arg1 arg2')


def main():
    """ 
    Main processing function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--output_file', '-o', type=str)
    parser.add_argument('--data', '-d', type=str)
    args = parser.parse_args()

    if args.data == 'GDA':
        abstracts, entities, relations = readPubTator(args, '|')
        type1 = ['Gene']
        type2 = ['Disease']
    elif args.data == 'CDR':
        abstracts, entities, relations = readPubTator(args, '|')
        # 判断应识别出的实体数量
        # num = 0
        # for i in entities:
        #     has = []
        #     for j in entities[i]:
        #         if (j.off1, j.off2) not in has:
        #             has.append((j.off1, j.off2))
        #             num += 1
        type1 = ['Chemical']
        type2 = ['Disease']
    elif args.data == 'BioRED':
        abstracts, entities, relations = readPubTator(args, ';')
        type1 = ['ChemicalEntity', 'GeneOrGeneProduct', 'SequenceVariant',
                 'ChemicalEntity', 'ChemicalEntity', 'ChemicalEntity', 'DiseaseOrPhenotypicFeature',
                 'DiseaseOrPhenotypicFeature', 'DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct', 'SequenceVariant',
                 'GeneOrGeneProduct', 'SequenceVariant']
        type2 = ['ChemicalEntity', 'GeneOrGeneProduct', 'SequenceVariant',
                 'DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct', 'SequenceVariant', 'GeneOrGeneProduct',
                 'SequenceVariant', 'ChemicalEntity', 'ChemicalEntity', 'ChemicalEntity',
                 'DiseaseOrPhenotypicFeature', 'DiseaseOrPhenotypicFeature']
    elif args.data == 'ChemProt' or args.data == 'DrugProt':
        abstracts, entities, relations = readPubTator(args, ';')
        type1 = ['CHEMICAL']
        type2 = ['GENE']
    else:
        print('Dataset non-existent.')
        sys.exit()

    if not os.path.exists(args.output_file + '_files/'):
        os.makedirs(args.output_file + '_files/')

    # Process
    positive, negative = 0, 0
    data = []
    with open(args.output_file, 'w') as data_out:
        pbar = tqdm(list(abstracts.keys()))
        for i in pbar:
            pbar.set_description("Processing Doc_ID {}".format(i))

            ''' Sentence Split '''
            orig_sentences = [item for sublist in [a.txt.split('\n') for a in abstracts[i]] for item in sublist]
            orig_sentence = orig_sentences[0]
            for indx, orig_sen in enumerate(orig_sentences):
                if indx != 0:
                    orig_sentence = orig_sentence + ' ' + orig_sen
            # 切分难以区分的实体
            # entities_list = entities[i]
            # se_list = []
            # for el in entities_list:
            #     se_list.append((el.off1, el.off2))
            # se_list = sorted(se_list, key=lambda x: (-x[1], -x[0]))
            # for se_list_id in range(1, len(se_list)):
            #     if se_list[se_list_id][1] == se_list[se_list_id - 1][0]:
            #         where = se_list[se_list_id][1]
            #         orig_sentence = orig_sentence[:where] + ' ' + orig_sentence[where:]
            orig_sentence = [orig_sentence]
            split_sents = sentence_split_genia(orig_sentence)
            split_sents = fix_sent_break(split_sents, entities[i])
            with open(args.output_file + '_files/' + i + '.split.txt', 'w') as f:
                f.write('\n'.join(split_sents))

            # adjust offsets
            new_entities = adjust_offsets(orig_sentences, split_sents, entities[i], show=False)

            ''' Tokenisation '''
            token_sents = tokenize_genia(split_sents)
            with open(args.output_file + '_files/' + i + '.split.tok.txt', 'w') as f:
                f.write('\n'.join(token_sents))

            # adjust offsets
            new_entities = adjust_offsets(split_sents, token_sents, new_entities, show=True)

            ''' Find mentions '''
            unique_entities = find_mentions(new_entities)
            with open(args.output_file + '_files/' + i + '.mention', 'wb') as f:
                pickle.dump(unique_entities, f, pickle.HIGHEST_PROTOCOL)

            if i in relations:
                pairs = generate_pairs(unique_entities, type1, type2, relations[i])
            else:
                pairs = generate_pairs(unique_entities, type1, type2, [])  # generate only negative pairs

            # 'pmid type arg1 arg2 dir cross'
            # data_out.write('{}\t{}'.format(i, '|'.join(token_sents)))
            json_pmid = i
            json_sents = [s.split(' ') for s in token_sents]
            json_entities = []
            for ens in unique_entities:
                en = []
                for e in unique_entities[ens]:
                    assert len(e.kb_id) == 1, '{}: {} <> {}'.format(json_pmid, len(e.kb_id), e.kb_id)
                    en.append({
                        "pos": [e.word_id[0], e.word_id[-1] + 1],
                        "type": e.type,
                        "sent_id": e.sent_no,
                        "name": e.name,
                        "type_id": e.kb_id[0]
                    })
                json_entities.append(en)
            json_labels = []
            for args_, p in pairs.items():
                if p.type == 'NR':
                    negative += 1
                    continue
                else:
                    positive += 1
                json_labels.append({
                    'h': p.arg1[0],
                    't': p.arg2[0],
                    'r': p.type,
                })
            json_data = {
                'title': json_pmid,
                'entities': json_entities,
                'labels': json_labels,
                'sents': json_sents
            }
            data.append(json_data)

            # for args_, p in pairs.items():
            #     if p.type != 'NR':
            #         positive += 1
            #     elif p.type == 'NR':
            #         negative += 1
            #
            #     data_out.write('\t{}\t{}\t{}\t{}-{}\t{}-{}'.format(p.type, p.dir, p.cross, p.closest[0].word_id[0],
            #                                                        p.closest[0].word_id[-1] + 1,
            #                                                        p.closest[1].word_id[0],
            #                                                        p.closest[1].word_id[-1] + 1))
            #
            #     data_out.write('\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            #         '|'.join([g for g in p.arg1]),
            #         '|'.join([e.name for e in unique_entities[p.arg1]]),
            #         unique_entities[p.arg1][0].type,
            #         ':'.join([str(e.word_id[0]) for e in unique_entities[p.arg1]]),
            #         ':'.join([str(e.word_id[-1] + 1) for e in unique_entities[p.arg1]]),
            #         ':'.join([str(e.sent_no) for e in unique_entities[p.arg1]])))
            #
            #     data_out.write('\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            #         '|'.join([g for g in p.arg2]),
            #         '|'.join([e.name for e in unique_entities[p.arg2]]),
            #         unique_entities[p.arg2][0].type,
            #         ':'.join([str(e.word_id[0]) for e in unique_entities[p.arg2]]),
            #         ':'.join([str(e.word_id[-1] + 1) for e in unique_entities[p.arg2]]),
            #         ':'.join([str(e.sent_no) for e in unique_entities[p.arg2]])))
            # data_out.write('\n')
        json.dump(data, data_out)
    print('Total positive pairs:', positive)
    print('Total negative pairs:', negative)


if __name__ == "__main__":
    main()
