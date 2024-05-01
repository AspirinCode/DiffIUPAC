import sentencepiece as spm
import sys
from collections import Counter

# file with a list of IUPAC names (can be just 1 line if you want)
#iupacs_fn = int(sys.argv[1])


with open("opsin_vocab_reduced.txt", "r") as f:
    words = f.read().split("\n")
words = list(map(str, range(100))) + words

smile_atom =[
        'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
        'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
        'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
        'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
        'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
        'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
        'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn',
        'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
        'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb',
        'Zn', 'Zr'
    ]
smile_non_atom = [
        '-', '=', '#', ':', '(', ')', '.', '[', ']', '+', '-', '\\', '/', '*',
        #'1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '@', 'AL', 'TH', 'SP', 'TB', 'OH',
    ]

#words = smile_atom+smile_non_atom+words

words = list(set(words))

vocab_size = len(words) + 1+100

user_defined_symbols = words

print("num user defined:", len(user_defined_symbols))

args = {"input": sys.argv[1],
        "model_type": "unigram",
        "model_prefix": "iupac_spm".format(vocab_size),
        "vocab_size": vocab_size,
        "input_sentence_size": 50000,
        "shuffle_input_sentence": True,
        "user_defined_symbols": user_defined_symbols,
        "split_by_number": False,
        "split_by_whitespace": False,
        "hard_vocab_limit": False,
        "max_sentencepiece_length": 320,
        "character_coverage": 0.99,
        "pad_id": 0,
        "eos_id": 1,
        "unk_id": 2,
        "bos_id": -1
        }
#"train_extremely_large_corpus": True

spm.SentencePieceTrainer.train(**args)
