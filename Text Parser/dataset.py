import os
import random

import cv2
import numpy as np
import torch
import json

from ruamel.yaml import YAML
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tokenizers import Tokenizer
from torchvision import transforms
from transformers import BertTokenizer, T5Tokenizer


# 定义数据集类
class GeoDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.tokenizer_sou = T5Tokenizer.from_pretrained(config['path2token'])
        self.tokenizer_tar = Tokenizer.from_file("vocab_tar_pre.json")
        self.vocab_tar = self.tokenizer_tar.get_vocab()
        self.ids, self.sources, self.targets = self.load_data()


    def load_data(self):
        prob_ens = []
        cdls = []
        ids = []
        if self.split == 'train_aug':
            with open(os.path.join(self.config['data_root'], 'train.json'), 'r', encoding='UTF-8') as f:
                data = json.load(f)
            with open(os.path.join(self.config['data_root'], f'{self.split}.json'), 'r', encoding='UTF-8') as f:
                aug = json.load(f)
        else:
            with open(os.path.join(self.config['data_root'], f'{self.split}.json'), 'r', encoding='UTF-8') as f:
                data = json.load(f)
        for id in data:
            if self.split == 'train_aug':
                random_values = random.sample(range(0, len(aug[id])), 3)
                for num in random_values:
                    prob_ens.append(aug[id][num])
                    cdl = 'text_cdl:' + data[id]['text_cdl'][0] + ';goal_cdl:' + data[id]['goal_cdl'][0]
                    cdls.append(cdl)
                    ids.append(id)
            else:
                prob_ens.append(data[id]['problem_text_en'])
                cdl = 'text_cdl:' + data[id]['text_cdl'][0] + ';goal_cdl:' + data[id]['goal_cdl'][0]
                cdls.append(cdl)
                ids.append(id)

        return ids, prob_ens, cdls

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        id = self.ids[idx]
        source = self.sources[idx]
        target = str(self.targets[idx])

        # Source text vectorization
        data = self.tokenizer_sou(source, padding='max_length', truncation=True, max_length=self.config['sou_maxlen'], return_tensors="pt")
        src_vec = data.input_ids.squeeze(dim=0)
        src_mask = data.attention_mask.squeeze(dim=0)
        soulen = src_mask.sum(dim=0)-1

        # Vectorization of the target text
        tar_list = self.tokenizer_tar.encode(target).ids
        tarlen = len(tar_list) + 2
        if tarlen <= self.config['tar_maxlen']:
            tar_fill = [self.vocab_tar['<start>']] + tar_list + [self.vocab_tar['<end>']] + [self.vocab_tar['<pad>']] * (self.config['tar_maxlen'] - tarlen)
            tar_mask = torch.cat((torch.ones(tarlen, dtype=torch.long), torch.zeros(self.config['tar_maxlen'] - tarlen, dtype=torch.long)))
        else:
            tar_fill = [self.vocab_tar['<start>']] + tar_list
            tar_fill = tar_fill[:self.config['tar_maxlen']-1] + [self.vocab_tar['<end>']]
            tar_mask = torch.ones(self.config['tar_maxlen'], dtype=torch.long)
        tar_vec = torch.LongTensor(tar_fill)

        if self.split=='train' or self.split == 'train_aug':
            return src_vec, src_mask, tar_vec, tar_mask
        else:
            return src_vec, src_mask, target, id
