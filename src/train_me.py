import argparse
import datetime
import os
import json
import torch
import warnings
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore")

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.model import MEModel
from src.evaluation import *
from src.data import read_dataset


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CDR", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--train_file", default='train.json', type=str)
    parser.add_argument("--dev_file", default='dev.json', type=str)
    parser.add_argument("--test_file", default='test.json', type=str)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--num_epoch", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--device", default="cuda:0", type=str, help="The running device.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--notes", default="", type=str)
    parser.add_argument("--no_dev", default=0, type=int)
    return parser.parse_args()


def train(args, model: MEModel, train_features, dev_features, test_features=None):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    result_dir = os.path.join(root_dir, f"result/{args.dataset}/me")
    model_dir = os.path.join(root_dir, f"model")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    log_file = os.path.join(result_dir, "{}.log".format(args.notes))
    f_log = open(log_file, 'a', encoding='utf-8')
    timestamp_start = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    f_log.write("Time start: {}\n".format(timestamp_start))
    f_log.write("{}\n".format(args))

    def finetune(features, optimizer, num_epoch, num_steps):
        best_dev_score, curr_test_score = -1, -1
        dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        train_iterator = range(num_epoch)
        total_steps = (len(dataloader) // args.gradient_accumulation_steps) * num_epoch
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        f_log.write("Total steps: {}\n".format(total_steps))
        print("Total steps: {}".format(total_steps))
        f_log.write("Warmup steps: {}\n".format(warmup_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            cum_loss = torch.tensor(0.0).to(args.device)
            for step, batch in tqdm(enumerate(dataloader), desc='train epoch {}'.format(epoch)):
                model.train()
                inputs = {"input_ids": batch["input_ids"].to(args.device),
                          "attention_mask": batch["attention_mask"].to(args.device),
                          "type_input_ids": batch["type_input_ids"].to(args.device),
                          "type_attention_mask": batch["type_attention_mask"].to(args.device),
                          "label": batch["label"].to(args.device)}
                loss = model.compute_loss(**inputs)
                loss = loss / args.gradient_accumulation_steps
                cum_loss += loss
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    cum_loss = torch.tensor(0.0).to(args.device)
                if step == len(dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0
                                                   and step % args.gradient_accumulation_steps == 0):
                    if args.no_dev:
                        test_score, test_coverage, test_output = evaluate(args, model, test_features, tag="test")
                        curr_test_score = test_score
                        f_log.write("epoch {}: test f1: {} | infer time: {}\n".format(epoch, test_score, test_coverage))
                        print("epoch {}: test f1: {} | infer time: {}".format(epoch, test_score, test_coverage))
                        test_file = os.path.join(result_dir, "{}_test.json".format(args.notes))
                        f_test = open(test_file, 'w', encoding='utf-8')
                        for entry in test_output:
                            jsonstr = json.dumps(entry)
                            f_test.write(jsonstr + "\n")
                        f_test.close()
                        continue
                    dev_score, dev_coverage, dev_output = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_coverage, test_output = evaluate(args, model, test_features, tag="test")
                    f_log.write(
                        "epoch {}: dev f1: {} | test f1: {} | infer time: {}\n".format(epoch, dev_score, test_score,
                                                                                       test_coverage))
                    print("epoch {}: dev f1: {} | test f1: {} | infer time: {}".format(epoch, dev_score, test_score,
                                                                                       test_coverage))
                    if dev_score > best_dev_score:
                        if args.save_path != "":
                            save_dir = os.path.dirname(args.save_path)
                            if not os.path.isdir(save_dir):
                                os.makedirs(save_dir)
                            torch.save(model.state_dict(), args.save_path)

                        best_dev_score = dev_score
                        dev_file = os.path.join(result_dir, "{}_dev.json".format(args.notes))
                        f_dev = open(dev_file, 'w', encoding='utf-8')
                        for entry in dev_output:
                            jsonstr = json.dumps(entry)
                            f_dev.write(jsonstr + "\n")
                        f_dev.close()

                        curr_test_score = test_score
                        test_file = os.path.join(result_dir, "{}_test.json".format(args.notes))
                        f_test = open(test_file, 'w', encoding='utf-8')
                        for entry in test_output:
                            jsonstr = json.dumps(entry)
                            f_test.write(jsonstr + "\n")
                        f_test.close()

        if args.no_dev:
            if args.save_path != "":
                save_dir = os.path.dirname(args.save_path)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), args.save_path)
            # copy test.log
            test_file = os.path.join(result_dir, "{}_test.json".format(args.notes))
            test_file_copy = os.path.join(result_dir,
                                          "{}_test_{}.json".format(args.notes, str(int(curr_test_score * 100000))))
            shutil.copy(test_file, test_file_copy)
            shutil.copy(os.path.join(model_dir, "{}.pt".format(args.notes)),
                        os.path.join(model_dir, "{}_{}.pt".format(args.notes, str(int(curr_test_score * 100000)))))
        else:
            # copy test.log & dev.log
            dev_file = os.path.join(result_dir, "{}_dev.json".format(args.notes))
            dev_file_copy = os.path.join(result_dir,
                                         "{}_dev_{}.json".format(args.notes, str(int(curr_test_score * 100000))))
            test_file = os.path.join(result_dir, "{}_test.json".format(args.notes))
            test_file_copy = os.path.join(result_dir,
                                          "{}_test_{}.json".format(args.notes, str(int(curr_test_score * 100000))))
            shutil.copy(dev_file, dev_file_copy)
            shutil.copy(test_file, test_file_copy)
            shutil.copy(os.path.join(model_dir, "{}.pt".format(args.notes)),
                        os.path.join(model_dir, "{}_{}.pt".format(args.notes, str(int(curr_test_score * 100000)))))

        f_log.write("best dev f1: {} | curr test f1: {}\n\n".format(best_dev_score, curr_test_score))
        print("best dev f1: {} | curr test f1: {}".format(best_dev_score, curr_test_score))
        return num_steps

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "bert" in n], },
        {"params": [p for n, p in model.named_parameters() if not "bert" in n], "lr": 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.num_epoch, num_steps)
    timestamp_end = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    f_log.write("Time end: {}\n\n".format(timestamp_end))
    f_log.close()


def evaluate(args, model: MEModel, features, tag=""):
    model.eval()
    dataloader = DataLoader(features, batch_size=args.test_batch_size,
                            shuffle=False, collate_fn=collate_fn)
    gold = []
    pred = []
    doc_maps = []
    doc_lens = []

    timestamp_start = datetime.datetime.now()
    for step, batch in tqdm(enumerate(dataloader)):
        gold.extend(batch["spans"])
        doc_maps.extend(batch["doc_map"])
        doc_lens.extend(batch["doc_len"])
        inputs = {"input_ids": batch["input_ids"].to(args.device),
                  "attention_mask": batch["attention_mask"].to(args.device),
                  "type_input_ids": batch["type_input_ids"].to(args.device),
                  "type_attention_mask": batch["type_attention_mask"].to(args.device)}
        with torch.no_grad():
            outputs = model.inference(**inputs)
            pred.extend(outputs)
    timestamp_end = datetime.datetime.now()

    # f1, p, r = compute_me_f1(pred, gold)
    output_spans = []
    # get span mapping
    tp_all, fp_all, fn_all = 0, 0, 0
    for ps, gs, doc_map, doc_len in zip(pred, gold, doc_maps, doc_lens):
        res = {'tp': [], 'fp': [], 'fn': []}
        # construct reverse doc map
        reversed_doc_map = {}
        for i_sent, sent in enumerate(doc_map):
            for word in sent.keys():
                reversed_doc_map[sent[word]] = [i_sent, word]
        for i in range(doc_len):
            if i not in reversed_doc_map:
                for j in range(i - 1, -1, -1):
                    if j in reversed_doc_map:
                        reversed_doc_map[i] = reversed_doc_map[j]
                        break
            if i not in reversed_doc_map:
                raise ValueError("Unexpected!")
        # be careful about the offset
        for s in ps:
            # need to convert back to original span
            if s in gs:
                res['tp'].append([reversed_doc_map[s[0] - 1], reversed_doc_map[s[1] - 1], s[2]])
            else:
                # 过滤预测的单词不全的情况
                if reversed_doc_map[s[0] - 1][0] == reversed_doc_map[s[1] - 1][0] \
                        and reversed_doc_map[s[0] - 1][1] == reversed_doc_map[s[1] - 1][1]:
                    continue
                res['fp'].append([reversed_doc_map[s[0] - 1], reversed_doc_map[s[1] - 1], s[2]])
        for s in gs:
            if s not in ps:
                res['fn'].append([reversed_doc_map[s[0] - 1], reversed_doc_map[s[1] - 1], s[2]])
        tp_all += len(res["tp"])
        fp_all += len(res["fp"])
        fn_all += len(res["fn"])
        res["tp"] = sorted(res["tp"], key=lambda x: x[0])
        res["fp"] = sorted(res["fp"], key=lambda x: x[0])
        res["fn"] = sorted(res["fn"], key=lambda x: x[0])
        res['stat'] = {'tp': len(res["tp"]), 'fp': len(res["fp"]), 'fn': len(res["fn"])}
        output_spans.append(res)
    coverage = timestamp_end - timestamp_start
    p = tp_all / (tp_all + fp_all + 1e-7)
    r = tp_all / (tp_all + fn_all + 1e-7)
    f1 = (p * r * 2) / (p + r + 1e-7)
    return f1, coverage, output_spans


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    type_max_len = max(len(f) for f in batch[0]["type_input_ids"])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    type_input_ids = [f + [0] * (type_max_len - len(f)) for f in batch[0]["type_input_ids"]]
    type_attention_mask = [[1.0] * len(f) + [0.0] * (type_max_len - len(f)) for f in batch[0]["type_input_ids"]]
    label = []
    for f in batch:
        la = []
        for i in range(len(f["label"])):
            la.append(f["label"][i] + [0] * (max_len - len(f["label"][i])))
        label.append(la)
    spans = [f["spans"] for f in batch]
    doc_map = [f["doc_map"] for f in batch]
    doc_len = [f["doc_len"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    type_input_ids = torch.tensor(type_input_ids, dtype=torch.long)
    type_attention_mask = torch.tensor(type_attention_mask, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    output = {"input_ids": input_ids, "attention_mask": attention_mask,
              "type_input_ids": type_input_ids, "type_attention_mask": type_attention_mask,
              "label": label, "spans": spans, "doc_map": doc_map, "doc_len": doc_len}
    return output


if __name__ == "__main__":
    args = get_opt()
    # get device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    # get config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if config.model_type == "bert":
        bos_token, eos_token, pad_token = "[CLS]", "[SEP]", "[PAD]"
    elif config.model_type == "roberta":
        bos_token, eos_token, pad_token = "<s>", "</s>", "<pad>"
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    config.bos_token_id, config.eos_token_id, config.pad_token_id = bos_token_id, eos_token_id, pad_token_id
    config.dataset = args.dataset
    # get model
    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    model = MEModel(config, encoder)
    model.to(device)

    if args.load_path == "":
        train_features = read_dataset(tokenizer, filename=args.train_file, dataset=args.dataset, task='me')
        dev_features = read_dataset(tokenizer, filename=args.dev_file, dataset=args.dataset, task='me')
        if args.no_dev:
            train_features.extend(dev_features)
        test_features = read_dataset(tokenizer, filename=args.test_file, dataset=args.dataset, task='me')
        train(args, model, train_features, dev_features, test_features)
    else:
        root_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.dirname(root_dir)
        result_dir = os.path.join(root_dir, f"result/{args.dataset}/me")
        os.makedirs(result_dir, exist_ok=True)
        dev_file = os.path.join(result_dir, "{}_dev.json".format(args.notes))
        test_file = os.path.join(result_dir, "{}_test.json".format(args.notes))

        model.load_state_dict(torch.load(args.load_path))
        test_features = read_dataset(tokenizer, filename=args.test_file, dataset=args.dataset, task='me')
        if args.no_dev == 0:
            dev_features = read_dataset(tokenizer, filename=args.dev_file, dataset=args.dataset, task='me')
            dev_score, dev_coverage, dev_output = evaluate(args, model, dev_features, tag="dev")
            print(dev_score, dev_coverage)
            with open(dev_file, 'w', encoding='utf-8') as f_dev:
                for entry in dev_output:
                    jsonstr = json.dumps(entry)
                    f_dev.write(jsonstr + "\n")
                f_dev.close()
        test_score, test_coverage, test_output = evaluate(args, model, test_features, tag="test")
        print(test_score, test_coverage)
        with open(test_file, 'w', encoding='utf-8') as f_test:
            for entry in test_output:
                jsonstr = json.dumps(entry)
                f_test.write(jsonstr + "\n")
            f_test.close()
