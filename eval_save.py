import sys
import pickle
import logging
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
)

import wandb
from tokenizer import *
from data import FunctionDataset_CL

from datautils.playdata import DatasetBase as DatasetBase

WANDB = True

def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=name)
    logger = logging.getLogger(__name__)
    s_handle = logging.StreamHandler(sys.stdout)
    s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logger.addHandler(s_handle)
    return logger

def finetune_eval(net, data_loader):
    net.eval()
    avg, gt, cons = [], [], []
    with torch.no_grad():
        for seq1, seq2, _, mask1, mask2, _ in tqdm(data_loader):
            input_ids1, attention_mask1 = seq1.cuda(), mask1.cuda()
            input_ids2, attention_mask2 = seq2.cuda(), mask2.cuda()

            anchor = net(input_ids=input_ids1, attention_mask=attention_mask1).pooler_output
            pos = net(input_ids=input_ids2, attention_mask=attention_mask2).pooler_output

            ans = 0
            for k in range(len(anchor)):
                vA = anchor[k:k+1].cpu()
                sim = []

                for j in range(len(pos)):
                    vB = pos[j:j+1].cpu()
                    sim_score = F.cosine_similarity(vA, vB).item()
                    sim.append(sim_score)
                    if j != k:
                        cons.append(sim_score)

                sim = np.array(sim)
                y = np.argsort(-sim)
                posi = np.where(y == k)[0][0] + 1
                gt.append(sim[k])
                ans += 1 / posi

            avg.append(ans / len(anchor))
            print("Current MRR:", np.mean(avg))

        final_mrr = np.mean(avg)
        with open("logft.txt", "a") as fi:
            print("MRR", final_mrr, file=fi)
        print("FINAL MRR:", final_mrr)
        return final_mrr

def eval(model, args, valid_set, logger):
    if WANDB:
        wandb.init(project='jTrans-finetune')
        wandb.config.update(args)

    device = torch.device("cuda")
    model.to(device)

    logger.info("Starting Evaluation...")
    valid_dataloader = DataLoader(valid_set, batch_size=args.eval_batch_size, num_workers=24, shuffle=True)
    mrr = finetune_eval(model, valid_dataloader)
    logger.info(f"Evaluation MRR: {mrr}")

    if WANDB:
        wandb.log({'mrr': mrr})

class BinBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings.position_embeddings = self.embeddings.word_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans Evaluation + Save EBDs")
    parser.add_argument("--model_path", type=str, default='./models/jTrans-finetune', help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default='./datautils/extract/', help="Path to the dataset")
    parser.add_argument("--output", type=str, default='./embeddings/Dataset-Muaz.pkl', help="Output path for experiment embeddings")
    parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer/')
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()

    now = datetime.now()
    log_name = f"jTrans-{args.model_path.replace('/','@')}-eval-{args.dataset_path.replace('/','@')}_savename_{args.output.replace('/','@')}.log"
    logger = get_logger(log_name)

    logger.info(f"Loading model from {args.model_path}")
    model = BinBertModel.from_pretrained(args.model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    logger.info("Tokenizer loaded")

    logger.info("Preparing dataset...")
    ft_valid_dataset = FunctionDataset_CL(
        tokenizer, args.dataset_path, None, True,
        opt=['O0', 'O1', 'O2', 'O3', 'Os'],
        add_ebd=True,
        convert_jump_addr=True,
        paired=args.paired
    )

    #logger.info("Generating embeddings for dataset...")
    #for i, func_data_str in tqdm(enumerate(ft_valid_dataset.datas)):
    #    ret = tokenizer([func_data_str],
    #                    add_special_tokens=True,
    #                    max_length=512,
    #                    padding='max_length',
    #                    truncation=True,
    #                    return_tensors='pt')
    #    input_ids, attention_mask = ret['input_ids'], ret['attention_mask']
    #    output = model(input_ids=input_ids, attention_mask=attention_mask)
    #    ft_valid_dataset.ebds[i] = output.pooler_output.detach().cpu()

    logger.info("Generating embeddings for dataset...")
    if args.paired:
        for i in tqdm(range(len(ft_valid_dataset.datas))):
            pairs = ft_valid_dataset.datas[i]
            for opt_level in ['O0', 'O1', 'O2', 'O3', 'Os']:
                idx = ft_valid_dataset.ebds[i].get(opt_level)
                if idx is not None:
                    ret = tokenizer([pairs[idx]], add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
                    input_ids, attention_mask = ret['input_ids'].cuda(), ret['attention_mask'].cuda()
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                    ft_valid_dataset.ebds[i][opt_level] = output.pooler_output.detach().cpu()
    else:
        for i, func_data in tqdm(enumerate(ft_valid_dataset.datas), total=len(ft_valid_dataset.datas)):
            ret = tokenizer([func_data], add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            input_ids, attention_mask = ret['input_ids'], ret['attention_mask']
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            ft_valid_dataset.ebds[i]["ebd"] = output.pooler_output.detach().cpu()


    logger.info("Saving embeddings to file...")
    with open(args.output, 'wb') as f:
        pickle.dump(ft_valid_dataset.ebds, f)

