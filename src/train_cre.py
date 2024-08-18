import argparse
import datetime
import os
import warnings
import shutil
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, Sampler
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from apex import amp
from src.model import CREModel
from src.evaluation import *
from src.data import *
from src.modules.sampler import CustomRandomSampler


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="docred", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--num_gcn_layers", default=3, type=int)
    parser.add_argument("--alpha", default=0.5, type=float, help="alpha parameter for cr/re loss ratio.")
    parser.add_argument("--beta", default=1.0, type=float, help="beta parameter for loss function.")
    parser.add_argument("--gamma", default=1.0, type=float, help="gamma parameter for cr/re loss ratio.")
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--train_file", default='train_gc.json', type=str)
    parser.add_argument("--dev_file", default='dev_gc.json', type=str)
    parser.add_argument("--test_file", default='test_gc.json', type=str)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--num_epoch", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--evaluation_start_epoch", default=-1, type=int,
                        help="Epoch that begin to evaluate.")
    parser.add_argument("--device", default="cuda:0", type=str, help="The running device.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--conv_num", default=1, type=int)
    parser.add_argument("--opn", type=str, default="mult", help="corr/sub/mult")
    parser.add_argument("--no_dev", default=1, type=int)
    parser.add_argument("--lambda1", default=2, type=int)
    parser.add_argument("--lambda2", default=1, type=int)
    parser.add_argument("--lambda3", default=2, type=int)
    parser.add_argument("--lambda4", default=2, type=int)
    return parser.parse_args()


def train(args, model, train_features, dev_features, test_features=None):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    result_dir = os.path.join(root_dir, f"result/{args.dataset}/cre")
    model_dir = os.path.join(root_dir, f"model")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    log_file = os.path.join(result_dir, "{}.log".format(args.notes))
    f_log = open(log_file, 'a', encoding='utf-8')
    timestamp_start = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    f_log.write("Time start: {}\n".format(timestamp_start))
    f_log.write("{}\n".format(args))

    def finetune(features, optimizer, num_epoch, num_steps):
        best_score, curr_test_score = -1, -1
        test_string_pre_best = ""
        sampler = CustomRandomSampler(features, train_batch_size=args.train_batch_size)
        if args.dataset == 'BioRED':
            dataloader = DataLoader(features, batch_size=args.train_batch_size, collate_fn=collate_fn, sampler=sampler)
        else:
            dataloader = DataLoader(features, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)
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
                          "type_input_list": batch["type_input_list"],
                          "cr_label": batch["cr_label"].to(args.device),
                          "re_label": batch["re_label"].to(args.device),
                          "cr_table_label": batch["cr_table_label"].to(args.device),
                          "re_table_label": batch["re_table_label"].to(args.device),
                          "spans": batch["spans"],
                          "hts": batch["hts"],
                          "graphs": batch["graphs"]}
                loss = model.compute_loss(**inputs)
                loss = loss / args.gradient_accumulation_steps
                cum_loss += loss
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    # if args.dataset != 'BioRED' or (step + 1) % 2 == 0:
                    #     model.zero_grad()
                    num_steps += 1
                    # wandb.log({"loss": cum_loss.item()}, step=num_steps)
                    cum_loss = torch.tensor(0.0).to(args.device)
                if step == len(dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0
                                                   and step % args.gradient_accumulation_steps == 0):
                    test_string_pre = ''
                    if args.no_dev:
                        test_score, test_output, test_string_pre_best = evaluate(args, model, test_features, tag="test")
                        curr_test_score = test_score
                        print("epoch {}: test f1: {}".format(epoch, test_output))
                        f_log.write("epoch {}: test f1: {}\n".format(epoch, test_output))
                        # torch.save(model.state_dict(), args.save_path)
                        continue
                    dev_score, dev_output, dev_string_pre = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output = 0, 0
                    if test_features is not None:
                        test_score, test_output, test_string_pre = evaluate(args, model, test_features, tag="test")
                    print("epoch {}: dev f1: {} | test f1: {}".format(epoch, dev_output, test_output))
                    f_log.write("epoch {}: dev f1: {} | test f1: {}\n".format(epoch, dev_output, test_output))
                    if dev_score > best_score:
                        best_score = dev_score
                        curr_test_score = test_score
                        test_string_pre_best = test_string_pre
                        if args.save_path != "":
                            save_dir = os.path.dirname(args.save_path)
                            if not os.path.isdir(save_dir):
                                os.makedirs(save_dir)
                            torch.save(model.state_dict(), args.save_path)
        if args.no_dev:
            if args.save_path != "":
                save_dir = os.path.dirname(args.save_path)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), args.save_path)
            shutil.copy(os.path.join(model_dir, "{}.pt".format(args.notes)),
                        os.path.join(model_dir, "{}_{}.pt".format(args.notes, str(int(curr_test_score * 100000)))))
        else:
            shutil.copy(os.path.join(model_dir, "{}.pt".format(args.notes)),
                        os.path.join(model_dir, "{}_{}.pt".format(args.notes, str(int(curr_test_score * 100000)))))

        f_log.write("best dev f1: {} | curr test f1: {}\n".format(best_score, curr_test_score))
        print("best dev f1: {} | curr test f1: {}".format(best_score, curr_test_score))
        # 输出结果看看
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        with open('./result/pred/' + str(args.dataset) + '_' + args.notes + '_' + str(timestamp) + '.txt', 'w') as file:
            file.write(test_string_pre_best)
        return num_steps

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "bert" in n], },
        {"params": [p for n, p in model.named_parameters() if not "bert" in n], "lr": 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.num_epoch, num_steps)
    timestamp_end = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    f_log.write("Time end: {}\n\n".format(timestamp_end))
    f_log.close()


def evaluate(args, model, features, tag=""):
    model.eval()
    # load vertexSet
    vertexSets = [f["vertexSet"] for f in features]
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    cr_gold, re_gold = [], []
    cr_pred, re_pred = [], []
    # infer time
    timestamp_inference_start = datetime.datetime.now()
    for step, batch in enumerate(dataloader):
        cr_gold.extend(batch["cr_clusters"])  # cr_clusters
        re_gold.extend(batch["re_triples"])
        inputs = {"input_ids": batch["input_ids"].to(args.device),
                  "attention_mask": batch["attention_mask"].to(args.device),
                  "type_input_ids": batch["type_input_ids"].to(args.device),
                  "type_attention_mask": batch["type_attention_mask"].to(args.device),
                  "type_input_list": batch["type_input_list"],
                  "spans": batch["spans"],
                  "graphs": batch["graphs"]}
        with torch.no_grad():
            outputs = model.inference(**inputs)
            for predictions in outputs['cr_predictions']:
                cr_pred.extend([predictions[0]])
            re_pred.extend(outputs['re_predictions'])
    timestamp_inference_end = datetime.datetime.now()
    timestamp_inference = timestamp_inference_end - timestamp_inference_start
    timestamp_inference = timestamp_inference.total_seconds()
    cr_f1, cr_p, cr_r = compute_cr_f1(pred_clusters=cr_pred, gold_clusters=cr_gold)  # hard f1
    avg, muc, b3, ceafe = compute_avg_cr_f1(pred_clusters=cr_pred, gold_clusters=cr_gold)  # avg f1
    # get cluster mapping
    cluster_mappings = []
    for pc, gc in zip(cr_pred, cr_gold):
        cluster_mappings.append(get_cluster_mapping(pc, gc)[1])
    re_f1, re_p, re_r, tp_cnt, pred_cnt, gold_cnt, tp_list = compute_re_f1_pubtator(cluster_mappings, re_pred,
                                                                                    re_gold, vertexSets)
    re_ign = .0
    # evaluate as real life
    cluster_mappings_real = []
    for pc, gc in zip(cr_pred, cr_gold):
        cluster_mappings_real.append(get_cluster_mapping_real(pc, gc)[1])
    re_f1_real, re_p_real, re_r_real, tp_cnt_real, tp_list_real \
        = compute_re_f1_pubtator_real(cluster_mappings_real, re_pred, re_gold, vertexSets)

    if args.dataset == 'CDR':
        id2rel = id2rel_cdr
    elif args.dataset == 'GDA':
        id2rel = id2rel_gda
    elif args.dataset == 'BioRED':
        id2rel = id2rel_biored
    else:
        id2rel = None
    string_pre = ""
    for pred_i, pred_list in enumerate(re_pred):
        for pred_one_re_i, pred_one_re in enumerate(pred_list):
            pred_h_cr_id = cr_pred[pred_i][pred_one_re['h']]
            pred_t_cr_id = cr_pred[pred_i][pred_one_re['t']]
            pred_h_cr = []
            pred_t_cr = []
            for pp in pred_h_cr_id:
                pred_h_cr.append(features[pred_i]['spans2entities'][pp])
            for pp in pred_t_cr_id:
                pred_t_cr.append(features[pred_i]['spans2entities'][pp])
            pred_r = id2rel[pred_one_re['r']]
            string_pre = string_pre + str(features[pred_i]['title']) + '\n' + str(pred_h_cr) + '\n' + str(
                pred_t_cr) + '\n' + str(pred_r) + ' ' + str(tp_list[pred_i][pred_one_re_i]) + ' ' + str(
                tp_list_real[pred_i][pred_one_re_i]) + '\n\n'

    string_pre = string_pre + "tp_cnt" + str(tp_cnt) + " tp_cnt_real" + str(tp_cnt_real) + " pred_cnt" + str(
        pred_cnt) + " gold_cnt" + str(gold_cnt)
    output_logs = {tag + "_infer_time": timestamp_inference,
                   tag + "_cr_f1": round(cr_f1 * 100, 4), tag + "_cr_p": round(cr_p * 100, 4),
                   tag + "_cr_r": round(cr_r * 100, 4), tag + "_re_f1": round(re_f1 * 100, 4),
                   tag + "_re_p": round(re_p * 100, 4), tag + "_re_r": round(re_r * 100, 4),
                   tag + "_re_f1_real": round(re_f1_real * 100, 4), tag + "_re_p_real": round(re_p_real * 100, 4),
                   tag + "_re_r_real": round(re_r_real * 100, 4),
                   tag + "_re_ign_f1": round(re_ign * 100, 4), tag + "_avg_f1": round(avg * 100, 4),
                   tag + "_muc_f1": round(muc * 100, 4), tag + "_b3_f1": round(b3 * 100, 4),
                   tag + "_ceafe_f1": round(ceafe * 100, 4)}
    return re_f1, output_logs, string_pre


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    type_max_len = max(len(f) for f in batch[0]["type_input_ids"])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    type_input_ids, type_attention_mask = [], []
    for type_ids in batch[0]["type_input_ids"]:
        type_input_ids.append(type_ids + [0] * (type_max_len - len(type_ids)))
        type_attention_mask.append([1.0] * len(type_ids) + [0.0] * (type_max_len - len(type_ids)))
    type_input_ids = torch.tensor(type_input_ids, dtype=torch.long)
    type_attention_mask = torch.tensor(type_attention_mask, dtype=torch.long)
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    spans = [f["spans"] for f in batch]
    type_input_list = [f["type_input_list"] for f in batch]
    type_num = type_input_ids.shape[0]
    cr_clusters = [f["cr_clusters"] for f in batch]
    re_triples = [f["re_triples"] for f in batch]
    if len(batch[0]["graphs"]) > 1:
        syntax_graph = [torch.tensor(f["graphs"]["syntax_graph"], dtype=torch.float) for f in batch]
        syntax_graph = torch.block_diag(*syntax_graph)
        mentions_type_graph = [torch.tensor(f["graphs"]["mentions_type_graph"], dtype=torch.float) for f in batch]
        mentions_type_graph = torch.block_diag(*mentions_type_graph)
        graphs = {"syntax_graph": syntax_graph, "mentions_type_graph": mentions_type_graph}
    else:
        graphs = {}

    if batch[0]["cr_label"] is None:
        cr_label = None
        re_label = None
        re_table_label = None
        cr_table_label = None
        hts = None
    elif batch[0]["re_label"] is None:
        cr_label = []
        for f in batch:
            cr_label.extend(f["cr_label"])
        cr_label = torch.tensor(cr_label)
        re_label = None
        re_table_label = None
        cr_table_label = None
        hts = None
    else:
        cr_label = []
        for f in batch:
            cr_label.extend(f["cr_label"])
        cr_label = torch.tensor(cr_label)
        re_label = [torch.tensor(f["re_label"]) for f in batch]  # in tensor form
        re_label = torch.cat(re_label, dim=0)
        hts = [f["hts"] for f in batch]

        cr_table_label = cr_label
        re_table_label = []
        for f in batch:
            re_table_label.extend(f["re_table_label"])
        re_table_label = torch.tensor(re_table_label)
    output = {"input_ids": input_ids, "attention_mask": attention_mask,
              "type_input_ids": type_input_ids, "type_attention_mask": type_attention_mask,
              "type_input_list": type_input_list,
              "spans": spans, "hts": hts,
              "cr_label": cr_label, "cr_clusters": cr_clusters,
              "re_label": re_label, "re_triples": re_triples,
              "graphs": graphs,
              "cr_table_label": cr_table_label, "re_table_label": re_table_label}
    return output


if __name__ == "__main__":
    args = get_opt()
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.dataset == 'BioRED':
        config.num_class = len(rel2id_biored)
    elif args.dataset == 'CDR':
        config.num_class = len(rel2id_cdr)
    elif args.dataset == 'GDA':
        config.num_class = len(rel2id_gda)
    else:
        raise ValueError("None dataset in rel2id")
    config.num_gcn_layers = args.num_gcn_layers
    config.alpha = args.alpha
    config.beta = args.beta
    config.gamma = args.gamma
    config.opn = args.opn
    config.conv_num = args.conv_num
    config.dropout = args.dropout
    config.adj_lambda = [args.lambda1, args.lambda2, args.lambda3, args.lambda4]
    if config.model_type == "bert":
        bos_token, eos_token, pad_token = "[CLS]", "[SEP]", "[PAD]"
    elif config.model_type == "roberta":
        bos_token, eos_token, pad_token = "<s>", "</s>", "<pad>"
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    config.bos_token_id, config.eos_token_id, config.pad_token_id = bos_token_id, eos_token_id, pad_token_id
    config.dataset = args.dataset
    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    model = CREModel(config, encoder)
    model.to(device)

    if args.load_path == "":
        train_features = read_dataset(tokenizer, filename=args.train_file, dataset=args.dataset, task='gc')
        if args.no_dev:
            dev_features = read_dataset(tokenizer, filename=args.dev_file, dataset=args.dataset, task='gc', no_dev=True)
            train_features.extend(dev_features)
        else:
            dev_features = read_dataset(tokenizer, filename=args.dev_file, dataset=args.dataset, task='gc')
        test_features = read_dataset(tokenizer, filename=args.test_file, dataset=args.dataset, task='gc')
        train(args, model, train_features, dev_features, test_features)
    else:
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        if args.no_dev == 0:
            dev_features = read_dataset(tokenizer, filename=args.dev_file, dataset=args.dataset, task='gc')
            dev_score, dev_output, dev_string_pre = evaluate(args, model, dev_features, tag="dev")
            print(dev_output)
        test_features = read_dataset(tokenizer, filename=args.test_file, dataset=args.dataset, task='gc')
        test_score, test_output, test_string_pre = evaluate(args, model, test_features, tag="test")
        print(test_output)
        os.makedirs('./result/pred', exist_ok=True)
        with open('./result/pred/test.txt', 'w') as file:
            file.write(test_string_pre)
