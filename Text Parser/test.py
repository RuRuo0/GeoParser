import argparse
import datetime
import logging
import time

from ruamel.yaml import YAML
import numpy as np
import random
from pathlib import Path

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu

from model import seq2seqModel
from utils import *
from dataset import GeoDataset
import warnings
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, filename='log/test-t5-base.log', filemode='w')

def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    references = list()
    hypotheses = list()
    ids = list()
    generate = {}

    with torch.no_grad():
        for i, (src_vec, src_mask, target, pro_id) in enumerate(data_loader):
            src_vec = src_vec.to(device)
            src_mask = src_mask.to(device)

            prediction = model.generate(config, src_vec, src_mask)

            for reference, hypothese in zip(target, prediction):
                references.append([reference])
                hypotheses.append(hypothese)

            assert len(references) == len(hypotheses)
            for item in pro_id:
                ids.append(item)

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

                dict = {'text_cdl':split_str(match_hyp.group(1).replace(' ', '')), 'goal_cdl':[match_hyp.group(2).replace(' ', '')]}
                generate[ids[i]] = dict

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
                formatError.append({'id': i, 'type': str(e)})

        with open(config['generate'], 'w') as file:
            json.dump(generate, file, indent=4)

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
                   "levelsnum_text": levelsnum_text, "levelsacc_text": levelsacc_text,
                   "perfectsnum_text": perfectsnum_text,
                   "perfectsacc_text": perfectsacc_text,
                   "levelsnum_goal": levelsnum_goal, "levelsacc_goal": levelsacc_goal,
                   "perfectsnum_goal": perfectsnum_goal,
                   "perfectsacc_goal": perfectsacc_goal}

    return bleu4, cdlsAcc, formatError, cdldict, cdlsPerfectAcc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='geo_config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating geo dataset")
    test_dataset = GeoDataset(config, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], pin_memory=True, drop_last=True)

    #### Model ####
    print("Creating model")
    checkpoint = torch.load(config['checkpoint'])
    model = checkpoint['model']
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best_cdlAcc = 0
    epochs_since_improvement = 0

    start_time = time.time()

    recent_bleu4, recent_cdlAcc, error, cdldict, r_cdlsPerfectAcc = evaluate(model, test_loader, device, config)

    print(f'The number of prediction format errors is:：{len(error)}')
    print(f'The type of prediction format errors is：{error}')
    logging.info(f'The number of prediction format errors is:：{len(error)}')

    print('Val BLEU-4: {bleu4:.3f}\t'
          'Val TextCdlAcc: {cdlAcc:.4f}\t'
          'Val CdlSPerfectAcc: {cdlsPerfectAcc:.4f}\t\n'
          .format(bleu4=recent_bleu4, cdlAcc=recent_cdlAcc, cdlsPerfectAcc=r_cdlsPerfectAcc))
    logging.info('Val BLEU-4: {bleu4:.3f}\t'
                 'Val TextCdlAcc: {cdlAcc:.4f}\t'
                 'Val CdlSPerfectAcc: {cdlsPerfectAcc:.4f}\t\n'
                 .format(bleu4=recent_bleu4, cdlAcc=recent_cdlAcc, cdlsPerfectAcc=r_cdlsPerfectAcc))

    with open(config['level'], 'w') as file:
        json.dump(cdldict, file, indent=4)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Epoch time {}'.format(total_time_str))
