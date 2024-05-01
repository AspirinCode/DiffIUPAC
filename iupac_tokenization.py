from transformers import (
    AdamW,
    DataCollatorWithPadding,
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
import os
import re
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import os.path as pt
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


class T5Collator:
    def __init__(self, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
    def __call__(self, records):
        # records is a list of dicts
        batch = {}
        padvals = {"input_ids": self.pad_token_id,'labels':-100}
        for k in records[0]:
            if k in padvals:
                batch[k] = pad_sequence([torch.tensor(r[k]) for r in records],
                                        batch_first=True,
                                        padding_value=padvals[k])
            else:
                batch[k] = torch.FloatTensor([r[k] for r in records]) #torch.Tensor
        return batch

class T5IUPACTokenizer(T5Tokenizer):
    def prepare_for_tokenization(self, text, is_split_into_words=False,
                                 **kwargs):
        return re.sub(" ", "_", text), kwargs

    def _decode(self, *args, **kwargs):
        # replace "_" with " ", except for the _ in extra_id_#
        text = super()._decode(*args, **kwargs)
        text = re.sub("extra_id_", "extraAidA", text)
        text = re.sub("_", " ", text)
        text = re.sub("extraAidA", "extra_id_", text)
        return text

    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1

    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))

    def _tokenize(self, text, sample=False):
        #pieces = super()._tokenize(text, sample=sample)
        pieces = super()._tokenize(text)
        # sentencepiece adds a non-printing token at the start. Remove it
        return ["<unk>"]+pieces[1:]


def get_iupac_tokenizer(is_train=1,full_path = './data'):

    iupac_tokenizer = T5IUPACTokenizer(vocab_file=pt.join(full_path,'iupac_spm.model'))
    iupac_vocab_size = iupac_tokenizer.vocab_size
    print('iupac_vocab_size:',iupac_vocab_size)
    if is_train:
        torch.save(iupac_tokenizer, pt.join(full_path,"real_iupac_tokenizer.pt"))
        print("training...",len(iupac_tokenizer))
    else:
        iupac_tokenizer = torch.load(pt.join(full_path,"real_iupac_tokenizer.pt"), map_location="cpu")
        print('fina_tune...',len(iupac_tokenizer))

    #collator = T5Collator(iupac_tokenizer.pad_token_id)

    return iupac_tokenizer

if __name__ == "__main__":

    iupac_tokenizer = get_iupac_tokenizer(is_train=1,full_path = './data')

    print(iupac_tokenizer,iupac_tokenizer.vocab_size)

    iupac_string = "2-(6-aminopurin-9-yl)-5-(methylsulfanylmethyl)oxolane-3,4-diol"
    iupac_encoded = iupac_tokenizer(iupac_string)
    iupac_merges = iupac_tokenizer.convert_ids_to_tokens(iupac_encoded["input_ids"])
    print(iupac_encoded)
    print(iupac_merges)

    line_number = 1

    valid_line=[]

    with open("data/pubchem_iupac.csv",'r') as f:
        myline = f.readline()
        while myline:
            #print("line_number:",line_number)

            iupac_encoded = iupac_tokenizer(myline)
            iupac_merges = iupac_tokenizer.convert_ids_to_tokens(iupac_encoded["input_ids"])
            #print(iupac_encoded)
            #print(iupac_merges)

            if iupac_encoded["input_ids"].count(2)==1:
                valid_line.append(myline)

            if line_number%50000==0:
                with open("data/pubchem_iupac_valid.csv",'a') as ff:
                    for j in valid_line:
                        ff.write(j)
                valid_line=[]

            myline = f.readline()
            line_number = 1+line_number

    

