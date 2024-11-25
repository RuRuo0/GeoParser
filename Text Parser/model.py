import warnings
warnings.filterwarnings("ignore")
import math
import torch
from torch import nn

from transformers import BertModel
from tokenizers import Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer


class seq2seqModel(nn.Module):
    def __init__(self, config):
        super(seq2seqModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(config['path2model'])

        self.model.lm_head = nn.Linear(self.model.lm_head.in_features, config['tar_vocab_size'], bias=False)

        self.tokenizer_tar = Tokenizer.from_file("vocab_tar_pre.json")
        self.vocab = self.tokenizer_tar.get_vocab()

    def forward(self, src_vec, src_mask, tar_vec, tar_mask):
        decoder_targets = tar_vec.masked_fill(tar_vec == self.vocab['<pad>'], -100)

        outputs = self.model(input_ids=src_vec,
                             attention_mask=src_mask,
                             # decoder_input_ids=tar_vec,
                             # decoder_attention_mask=tar_mask,
                             labels=decoder_targets)

        return outputs.loss

    def generate(self, config, src_vec, src_mask):
        outputs = self.model.generate(input_ids=src_vec,
                                      attention_mask=src_mask,
                                      max_length=config['tar_maxlen'],
                                      num_beams=config['num_beams'],
                                      early_stopping=True,
                                      bos_token_id=self.vocab['<start>'],
                                      pad_token_id=self.vocab['<pad>'],
                                      eos_token_id=self.vocab['<end>'],
                                      repetition_penalty=config['repetition_penalty']
                                      )

        predictions = []
        for output in outputs:
            pre_vec = [w for w in output.tolist() if
                       w not in {self.vocab['<start>'], self.vocab['<end>'], self.vocab['<pad>']}]
            prediction = self.tokenizer_tar.decode(pre_vec)
            predictions.append(prediction)
        return predictions