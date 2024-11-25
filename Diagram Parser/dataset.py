import os
import cv2
import numpy as np
import torch
import json
from ruamel.yaml import YAML
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tokenizers import Tokenizer
from torchvision import transforms
from data_augment import augment_data


class GeoDataset(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.tokenizer = Tokenizer.from_file("vocab.json")
        self.image_paths, self.captions = self.load_data()

        self.transform = transform

    def load_data(self):
        image_paths = []
        # cs_cdls = []
        cdls = []
        if self.split == 'train_aug':
            with open(os.path.join(self.config['data_root'], 'train.json'), 'r', encoding='UTF-8') as f:
                data = json.load(f)
            data = augment_data(data,2)
        else:
            with open(os.path.join(self.config['data_root'], f'{self.split}.json'), 'r', encoding='UTF-8') as f:
                data = json.load(f)
        for id in data:
            if self.split == 'train_aug':
                for num in data[id]:
                    image_paths.append(os.path.join(self.config['data_root'], 'train', f'{id}.png'))
                    # cs_cdls.append(data[id][num]['image_cdl'])
                    cdl = 'construction_cdl:' + data[id][num]['construction_cdl'][0] + ';image_cdl:' + data[id][num]['image_cdl'][0]
                    cdls.append(cdl)

            else:
                image_paths.append(os.path.join(self.config['data_root'], self.split, f'{id}.png'))
                # cs_cdls.append(data[id]['image_cdl'])
                cdl = 'construction_cdl:' + data[id]['construction_cdl'][0] + ';image_cdl:' + data[id]['image_cdl'][0]
                cdls.append(cdl)

        return image_paths, cdls

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_id = image_path.split('\\')[-1].split('.')[0]
        caption = str(self.captions[idx])

        # Load the image
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (self.config['image_size'], self.config['image_size']))
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img)
        img = img / 255.
        img = self.transform(img)

        # Text vectorization
        output = self.tokenizer.encode(caption)
        output_list = output.ids
        caplen = len(output) + 2
        start_id = self.tokenizer.get_vocab()['<start>']
        end_id = self.tokenizer.get_vocab()['<end>']
        pad_id = self.tokenizer.get_vocab()['<pad>']
        if caplen <= self.config['max_length']:
            output_fill = [start_id] + output_list + [end_id] + [pad_id] * (self.config['max_length'] - caplen)
            capmask = torch.cat((torch.ones(caplen, dtype=torch.long), torch.zeros(self.config['max_length'] - caplen, dtype=torch.long)))
        else:
            output_fill = [start_id] + output_list
            output_fill = output_fill[:self.config['max_length']-1] + [end_id]
            capmask = torch.ones(self.config['max_length'], dtype=torch.long)
        capvec = torch.LongTensor(output_fill)
        caplen = torch.sum(capmask)

        if self.split == 'train' or self.split == 'train_aug':
            return img, caption, capvec, capmask, img_id, caplen
        else:
            return img, caption, capvec, img_id, caplen

if __name__ == '__main__':
    yaml = YAML(typ='rt')
    config = yaml.load(open('geo_config.yaml', 'r'))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    dataset = {split: GeoDataset(config, split, transform=transforms.Compose([normalize]))
                   for split in ['train', 'val', 'test']}
    train_loader = DataLoader(dataset['train'], batch_size=config['batch_size'], shuffle=True, pin_memory=True, drop_last = True)
    lens = []
    for i, (image, caption, capvec,capmask, id) in enumerate(train_loader):
        lens.append(len(caption[0]))
    print(sum(lens)/len(lens))
    print(max(lens),min(lens))
    lens = sorted(lens)
    print(lens)