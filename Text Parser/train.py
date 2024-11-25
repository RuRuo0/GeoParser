'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import datetime
import logging
import time

from ruamel.yaml import YAML
import numpy as np
import random

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu

from model import seq2seqModel
from utils import *
from dataset import GeoDataset
import warnings
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, filename='log/t5-base.log', filemode='w')

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()
    losses = []

    print_freq = 1000

    for i, (src_vec, src_mask, tar_vec, tar_mask) in enumerate(data_loader):
        src_vec = src_vec.to(device)
        src_mask = src_mask.to(device)
        tar_vec = tar_vec.to(device)
        tar_mask = tar_mask.to(device)

        loss = model(src_vec, src_mask, tar_vec, tar_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Loss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))

    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    references = list()
    hypotheses = list()
    ids = list()

    with torch.no_grad():
        for i, (src_vec, src_mask, target, pro_id) in enumerate(data_loader):
            src_vec = src_vec.to(device)
            src_mask = src_mask.to(device)

            prediction = model.generate(config, src_vec, src_mask)

            for id in pro_id:
                ids.append(id)
            for reference, hypothese in zip(target, prediction):
                references.append([reference])
                hypotheses.append(hypothese)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)

        formatError = []
        cdlsAccs = []
        levelsnum_text = [0] * 20
        levelsacc_text = [0] * 20
        levelsnum_goal = [0] * 2
        levelsacc_goal = [0] * 2
        levelsnum = [0] * 20
        levelsacc = [0] * 20
        perfectsnum_text = [0] * 20
        perfectsnum_goal = [0] * 20
        perfectsnum = [0] * 20
        for i in range(len(references)):
            reference = references[i][0]
            hypothese = hypotheses[i]
            try:
                match_ref = re.search(r"text_cdl:(.*?);goal_cdl:(.*)", reference)
                match_hyp = re.search(r"text_cdl:(.*?);goal_cdl:(.*)", hypothese)
                acc_text, level_text = getTextCdlAcc(ids[i], match_ref.group(1), match_hyp.group(1))
                acc_goal, level_goal = getGoalCdlAcc(match_ref.group(2), match_hyp.group(2))
                acc = (acc_text * level_text + acc_goal * level_goal) / (level_text + level_goal)

                levelsnum_text[level_text - 1] += 1
                levelsacc_text[level_text - 1] += acc_text
                levelsnum_goal[level_goal - 1] += 1
                levelsacc_goal[level_goal - 1] += acc_goal
                levelsnum[(level_text + level_goal) - 1] += 1
                levelsacc[(level_text + level_goal) - 1] += acc
                if acc_text == 1:
                    perfectsnum_text[level_text - 1] += 1
                if acc_goal == 1:
                    perfectsnum_goal[level_goal - 1] += 1
                if acc == 1:
                    perfectsnum[(level_text + level_goal) - 1] += 1
                cdlsAccs.append(acc)
            except Exception as e:
                # 捕获通用异常，并自定义处理
                cdlsAccs.append(0)
                formatError.append({'id':i, 'type':str(e)})
        cdlsAcc = sum(cdlsAccs) / len(cdlsAccs)
        cdlsPerfectAcc = sum(perfectsnum) / 1050
        levelsacc = getAccAvg(levelsacc, levelsnum)
        perfectsacc = getAccAvg(perfectsnum, levelsnum)
        levelsacc_text = getAccAvg(levelsacc_text, levelsnum_text)
        perfectsacc_text = getAccAvg(perfectsnum_text, levelsnum_text)
        levelsacc_goal = getAccAvg(levelsacc_goal, levelsnum_goal)
        perfectsacc_goal = getAccAvg(perfectsnum_goal, levelsnum_goal)
        cdldict = {"levelsnum": levelsnum, "levelsacc": levelsacc, "perfectsnum": perfectsnum,
                   "perfectsacc": perfectsacc,
                   "levelsnum_text": levelsnum_text, "levelsacc_text": levelsacc_text, "perfectsnum_text": perfectsnum_text,
                   "perfectsacc_text": perfectsacc_text,
                   "levelsnum_goal": levelsnum_goal, "levelsacc_goal": levelsacc_goal, "perfectsnum_goal": perfectsnum_goal,
                   "perfectsacc_goal": perfectsacc_goal}

    return bleu4, cdlsAcc, formatError, cdldict, cdlsPerfectAcc


def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating geo dataset")
    val_dataset = GeoDataset(config, 'val')
    test_dataset = GeoDataset(config, 'test')
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], pin_memory=True, drop_last=True)

    #### Model #### 
    print("Creating model")
    model = seq2seqModel(config)
    # checkpoint = torch.load(config['checkpoint'])
    # model = checkpoint['model']
    # epoch_cur = checkpoint['epoch'] + 1
    model = model.to(device)   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best_cdlAcc = 0
    epochs_since_improvement = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        # generate a random number of seeds
        random_seed = random.randint(0, 2 ** 32 - 1)
        random.seed(random_seed)
        # augment training dataset online
        train_dataset = GeoDataset(config, 'train')
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, drop_last=True)

        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        # epoch += epoch_cur

        train_stats = train(model, train_loader, optimizer, epoch, device)

        recent_bleu4, recent_cdlsAcc, error, cdldict, r_cdlsPerfectAcc = evaluate(model, val_loader, device, config)

        print('\nEpoch: [{0}]\t'
              'Train Loss: {train_loss:.3f}\t'
              'Val BLEU-4: {bleu4:.3f}\t'
              'Val CdlAcc: {cdlAcc:.4f}\t'
              'Val CdlSPerfectAcc: {cdlsPerfectAcc:.4f}\t\n'
              .format(epoch, train_loss=train_stats, bleu4=recent_bleu4, cdlAcc=recent_cdlsAcc, cdlsPerfectAcc=r_cdlsPerfectAcc))
        logging.info('Epoch: [{0}]\t'
                     'Train Loss: {train_loss:.3f}\t'
                     'Val BLEU-4: {bleu4:.3f}\t'
                     'Val CdlAcc: {cdlAcc:.4f}\t'
                     'Val CdlSPerfectAcc: {cdlsPerfectAcc:.4f}\t\n'
                     .format(epoch, train_loss=train_stats, bleu4=recent_bleu4, cdlAcc=recent_cdlsAcc, cdlsPerfectAcc=r_cdlsPerfectAcc))
        print(f'The number of prediction format errors is：{len(error)}')
        logging.info(f'The number of prediction format errors is：{len(error)}')

        # Check if there was an improvement
        is_best = recent_cdlsAcc > best_cdlAcc
        best_cdlAcc = max(recent_cdlsAcc, best_cdlAcc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            logging.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0


        save_checkpoint(config['ckpt_name'], epoch, epochs_since_improvement, model, optimizer, recent_bleu4, recent_cdlsAcc, is_best)

        if epochs_since_improvement > 3:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Epoch time {}'.format(total_time_str))
    logging.info('Epoch time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='geo_config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    yaml = YAML(typ='rt')

    config = yaml.load(open(args.config, 'r'))

    main(args, config)