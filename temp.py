


import pandas as pd


a=pd.read_csv('./data/pubchem_MolWt200-1000_iupac.csv',sep='|',header=0)


#mask = (a['PUBCHEM_IUPAC_NAME'].str.len() <=512)

#a = a.loc[mask]

#a=a[a.PUBCHEM_IUPAC_NAME.astype(str).str.len()<=512]
from iupac_tokenization import get_iupac_tokenizer
from smile_tokenization import get_smiles_tokenizer

iupac_tokenizer  = get_iupac_tokenizer(is_train=1,full_path ='./data')
smiles_tokenizer = get_smiles_tokenizer(is_train=1,checkpoint = "./data/smile_tocken")

def seq_valid_iupac(iupac_tokenizer,myline):
	iupac_encoded = iupac_tokenizer(myline)
	if iupac_encoded["input_ids"].count(2)==1:
		return 1
	else:
		return 0

def seq_valid_smiles(smiles_tokenizer,myline):
	iupac_encoded = smiles_tokenizer(myline)
	if iupac_encoded["input_ids"].count(1)==1:
		return 1
	else:
		return 0

a['PUBCHEM_IUPAC_NAME_if'] = a['PUBCHEM_IUPAC_NAME'].apply(lambda x :seq_valid_iupac(iupac_tokenizer,x))
a['canon_smiles_if'] = a['canon_smiles'].apply(lambda x :seq_valid_smiles(smiles_tokenizer,x))

a = a[(a['PUBCHEM_IUPAC_NAME_if']==1)&(a['canon_smiles_if']==1)]

#a[['PUBCHEM_IUPAC_NAME']].to_csv('./data/pubchem_iupac_valid.csv',header=None,index=None,sep='|')
#a[['canon_smiles']].to_csv('./data/pubchem_smiles_valid.csv',header=None,index=None,sep='|')


b=a.iloc[0:30000000]

b[['PUBCHEM_IUPAC_NAME']].to_csv('./data/pubchem_iupac_train_3qw.csv',header=None,index=None,sep='|')
b[['canon_smiles']].to_csv('./data/pubchem_smiles_train_3qw.csv',header=None,index=None,sep='|')

c=a.iloc[30000000:]

c[['PUBCHEM_IUPAC_NAME']].to_csv('./data/pubchem_iupac_valid_3qw.csv',header=None,index=None,sep='|')
c[['canon_smiles']].to_csv('./data/pubchem_smiles_valid_3qw.csv',header=None,index=None,sep='|')
