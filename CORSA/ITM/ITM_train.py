import os
import logging
import argparse
import random
import datetime
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from itertools import cycle
from transformers import RobertaTokenizer, RobertaModel

from torch import optim
from torch.nn import CrossEntropyLoss

from DataProcessor import *
from model import Coarse2Fine

from optimization import BertAdam

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokens, input_ids, input_mask, relation_label, img_feat = batch

    input_ids = list(map(list, zip(*input_ids)))
    input_mask = list(map(list, zip(*input_mask)))

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    relation_label = relation_label.to(device).long()

    img_feat = img_feat.to(device).float()


    return tokens, input_ids, input_mask, relation_label, img_feat


def main():
    VG_global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vg_data_dir='/data/liuxj/aspect_sentiment_detect/AOM(itm+object_detect)/src/data/twitter2015_new'
    vg_image_feat='/data/liuxj/aspect_sentiment_detect/ITM-main/data/twitter_images/twitter2015_extract'

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_dataset_VG = MyDataset(vg_data_dir+'/train.json', vg_image_feat, tokenizer,
                                 max_seq_len=128, num_roi_boxes=100)
    train_dataloader_VG = Data.DataLoader(dataset=train_dataset_VG, shuffle=True, batch_size=64,
                                          num_workers=0)

    train_number = train_dataset_VG.number
    num_train_steps = int(train_number / 64* 10)

    model = Coarse2Fine(roberta_name='robertabase', roi_num=100)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_VG = BertAdam(optimizer_grouped_parameters,
                            lr=1,
                            warmup=0.1,
                            t_total=num_train_steps)

    print("******** Running training ********")
    for train_idx in trange(int(100), desc="Epoch"):
        print("*********Epoch: " + str(train_idx) + " *********")
        ### train
        model.train()
        acc_list=[]
        loss_list=[]
        for batch in train_dataloader_VG:
            #print(batch)
            tokens, input_ids, input_mask, relation_label, img_feat = post_dataloader(batch)
            pred_loss, pred_score,acc= model(input_ids=input_ids,input_mask=input_mask,
                                        img_feat=img_feat,relation_label=relation_label,
                                        pred_loss_ratio=1.)
            acc_list.append(acc)
            loss_list.append(pred_loss)
            #print('LOSS:',pred_loss)
            #print('ACC:', acc)
            loss_VG = pred_loss
            loss_VG.backward()

            lr_this_step = 1 * warmup_linear(VG_global_step / num_train_steps,0.1)
            for param_group in optimizer_VG.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_VG.step()
            optimizer_VG.zero_grad()
        train_acc=sum(acc_list)/len(acc_list)
        train_loss = sum(loss_list) / len(loss_list)
        print('train_ACC:', train_acc)
        print('train_LOSS:', train_loss)

main()
