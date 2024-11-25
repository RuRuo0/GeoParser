import argparse
import datetime
import logging
import time
from ruamel.yaml import YAML
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu
from blip import blip_decoder
from utils import *
from dataset import GeoDataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, filename='log/train.log', filemode='w')

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()
    losses = []
    print_freq = 1000

    for i, (image, _, capvec, capmask, img_id, _) in enumerate(data_loader):
        image = image.to(device)
        capvec = capvec.to(device)
        capmask = capmask.to(device)

        loss = model(image, capvec, capmask)
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
        for i, (image, target, _, image_id, _) in enumerate(data_loader):

            image = image.to(device)
            captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                      min_length=config['min_length'])

            for reference, hypothese in zip(target, captions):
                references.append([reference.replace(' ', '')])
                hypotheses.append(hypothese.replace(' ', ''))

            assert len(references) == len(hypotheses)

            for id in image_id:
                ids.append(id.split('/')[-1])
        bleu4 = corpus_bleu(references, hypotheses)

        formatError = []
        cdlsAccs = []
        levelsnum_cs = [0] * 20
        levelsacc_cs = [0] * 20
        levelsnum_img = [0] * 20
        levelsacc_img = [0] * 20
        levelsnum = [0] * 35
        levelsacc = [0] * 35
        perfectsnum_cs = [0] * 20
        perfectsnum_img = [0] * 20
        perfectsnum = [0] * 35

        for i in range(len(references)):
            reference = references[i][0]
            hypothese = hypotheses[i]
            try:
                match_ref = re.search(r"construction_cdl:(.*?);image_cdl:(.*)", reference)
                match_hyp = re.search(r"construction_cdl:(.*?);image_cdl:(.*)", hypothese)

                acc_cs, level_cs = getConsCdlAcc(match_ref.group(1), match_hyp.group(1))
                acc_img, level_img = getImgCdlAcc(ids[i], match_ref.group(2), match_hyp.group(2))
                acc = (acc_cs * level_cs + acc_img * level_img) / (level_cs + level_img)

                levelsnum_cs[level_cs - 1] += 1
                levelsacc_cs[level_cs - 1] += acc_cs
                levelsnum_img[level_img - 1] += 1
                levelsacc_img[level_img - 1] += acc_img
                levelsnum[(level_cs + level_img) - 1] += 1
                levelsacc[(level_cs + level_img) - 1] += acc
                if acc_cs == 1:
                    perfectsnum_cs[level_cs - 1] += 1
                if acc_img == 1:
                    perfectsnum_img[level_img - 1] += 1
                if acc == 1:
                    perfectsnum[(level_cs + level_img) - 1] += 1
                cdlsAccs.append(acc)
            except Exception as e:
                # Catch common exceptions and customize handling
                cdlsAccs.append(0)
                formatError.append({'id':i,'type':str(e)})
        cdlsAcc = sum(cdlsAccs) / len(cdlsAccs)
        cdlsPerfectAcc = sum(perfectsnum) / (len(data_loader) * config['batch_size'])
        levelsacc = getAccAvg(levelsacc, levelsnum)
        perfectsacc = getAccAvg(perfectsnum, levelsnum)
        levelsacc_cs = getAccAvg(levelsacc_cs, levelsnum_cs)
        perfectsacc_cs = getAccAvg(perfectsnum_cs, levelsnum_cs)
        levelsacc_img = getAccAvg(levelsacc_img, levelsnum_img)
        perfectsacc_img = getAccAvg(perfectsnum_img, levelsnum_img)
        cdldict = {"levelsnum": levelsnum, "levelsacc": levelsacc, "perfectsnum": perfectsnum,"perfectsacc": perfectsacc,
                   "levelsnum_cs": levelsnum_cs, "levelsacc_cs": levelsacc_cs, "perfectsnum_cs": perfectsnum_cs,"perfectsacc_cs": perfectsacc_cs,
                   "levelsnum_img": levelsnum_img, "levelsacc_img": levelsacc_img, "perfectsnum_img": perfectsnum_img,"perfectsacc_img": perfectsacc_img}

    return bleu4, cdlsAcc, formatError, cdldict, cdlsPerfectAcc

def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    ##### Dataset #####
    print("Creating GeoDataset")
    # logging.info("Creating GeoDataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Creating validation and test set
    val_dataset = GeoDataset(config, 'val', transform=transforms.Compose([normalize]))
    test_dataset = GeoDataset(config, 'test', transform=transforms.Compose([normalize]))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], pin_memory=True, drop_last=True)

    ##### Model #####
    print("Creating model")
    logging.info("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    # checkpoint = torch.load(config['checkpoint'])
    # model.load_state_dict(checkpoint['model'])
    # model = model.to(device)
    # epoch_cur = checkpoint['epoch'] + 1

    model = model.to(device)
    model_without_ddp = model
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best_cdlAcc = 0
    epochs_since_improvement = 0

    print("Start training")
    logging.info("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        # generate a random number of seeds
        random_seed = random.randint(0, 2 ** 32 - 1)
        random.seed(random_seed)
        # augment training dataset online
        train_dataset = GeoDataset(config, 'train', transform=transforms.Compose([normalize]))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,drop_last=True)
        # training...
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        # epoch += epoch_cur
        train_stats = train(model, train_loader, optimizer, epoch, device)

        # valiating...
        recent_bleu4, recent_cdlsAcc, error, cdldict, r_cdlsPerfectAcc = evaluate(model_without_ddp, test_loader, device, config)

        print('\nEpoch: [{0}]\t'
              'Train Loss: {train_loss:.3f}\t'
              'Val BLEU-4: {bleu4:.3f}\t'
              'Val CdlSAcc: {cdlsAcc:.4f}\t\n'
              'Val CdlSPerfectAcc: {cdlsPerfectAcc:.4f}\t\n'
              .format(epoch, train_loss=train_stats, bleu4=recent_bleu4, cdlsAcc=recent_cdlsAcc, cdlsPerfectAcc=r_cdlsPerfectAcc))
        logging.info('Epoch: [{0}]\t'
                     'Train Loss: {train_loss:.3f}\t'
                     'Val BLEU-4: {bleu4:.3f}\t'
                     'Val CdlSAcc: {cdlsAcc:.4f}\t\n'
                     'Val CdlSPerfectAcc: {cdlsPerfectAcc:.4f}\t\n'
                     .format(epoch, train_loss=train_stats, bleu4=recent_bleu4, cdlsAcc=recent_cdlsAcc, cdlsPerfectAcc=r_cdlsPerfectAcc))
        print(f'Numbers of  predicting format errors：{len(error)}')
        logging.info(f'Numbers of  predicting format errors：{len(error)}')

        is_best = recent_cdlsAcc > best_cdlAcc
        best_cdlAcc = max(recent_cdlsAcc, best_cdlAcc)
        if not is_best:
            epochs_since_improvement += 1
            print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
            logging.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            with open(config['level'], 'w') as file:
                json.dump(cdldict, file, indent=4)

        save_checkpoint(config['ckpt_name'], epoch, epochs_since_improvement, model, optimizer, recent_bleu4, recent_cdlsAcc, is_best)

        if epochs_since_improvement >= 4:
            print("early_stopping")
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