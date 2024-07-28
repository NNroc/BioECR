import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='./data/GDA/training.pubtator', type=str)
parser.add_argument('--output_train', default='./data/GDA/train.pubtator', type=str)
parser.add_argument('--output_dev', default='./data/GDA/dev.pubtator', type=str)
parser.add_argument('--list', default='./data/GDA/train_gda_docs', type=str)
args = parser.parse_args()

with open(args.list, 'r') as infile:
    docs = [i.rstrip() for i in infile]
docs = frozenset(docs)

pmid = -1
with open(args.input_file, 'r') as infile, open(args.output_train, 'w') as otr, open(args.output_dev, 'w') as odev:
    for line in infile:
        if line == '\n':
            if pmid in docs:
                otr.write(line)
            else:
                odev.write(line)
            pmid = -1
            continue
        if pmid == -1:
            pmid = line.rstrip().split('|')[0]
        if pmid in docs:
            otr.write(line)
        else:
            odev.write(line)
