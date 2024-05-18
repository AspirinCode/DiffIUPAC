[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](https://github.com/AspirinCode/DiffIUPAC)


## DiffIUPAC

**Diffusion-based generative drug-like molecular editing with chemical natural language**  



![Model Architecture of DiffIUPAC](https://github.com/AspirinCode/DiffIUPAC/blob/main/figure/framework_figure.png)


## Acknowledgements
We thank the authors of C5T5: Controllable Generation of Organic Molecules with Transformers, IUPAC2Struct: Transformer-based artificial neural networks for the conversion between chemical notations, Deep molecular generative model based on variant transformer for antiviral drug design, and SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers for releasing their code. The code in this repository is based on their source code release (https://github.com/dhroth/c5t5, https://github.com/sergsb/IUPAC2Struct, https://github.com/AspirinCode/TransAntivirus, and https://github.com/yuanhy1997/seqdiffuseq). If you find this code useful, please consider citing their work.


## Requirements
```python
conda create -n diffiupac python=3.8
conda install mpi4py
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0
pip install -r requirements.txt

```

https://github.com/rdkit/rdkit  




## System Requirerments
*  requires system memory larger than 228GB.  

*  (if GPU is available) requires GPU memory larger than 80GB.  




## Data


**PubChem**
https://pubchem.ncbi.nlm.nih.gov/


## Training

To run the code, we use iwslt14 en-de as an illustrative example:

**Prepare the data:** 
Learning the BPE tokenizer by
```
sh ./tokenizer_utils.py train-byte-level iwslt14 10000 
```

**To train with the following line:**  
```
mkdir ckpts
bash ./train_scripts/train.sh 0 iupac smiles
#(for en to de translation) bash ./train_scripts/iwslt_en_de.sh 0 smiles iupac 
```

You may modify the scripts in ./train_scripts for your own training settings.


**To fine tune with the following line:**  

```
bash ./train_scripts/fine_tune.sh 0 iupac smiles

```

## Generating

To run the code, example data is in the example folder:

```
bash ./train_scripts/gen_opt.sh

```


## Model Metrics

### MOSES

Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.  
https://github.com/molecularsets/moses  

### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness  

https://github.com/ohuelab/QEPPI  

## License
Code is released under GNU GENERAL PUBLIC LICENSE.


## Cite:
* Yuan, Hongyi, Zheng Yuan, Chuanqi Tan, Fei Huang, and Songfang Huang. "SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers." arXiv preprint arXiv:2212.10325 (2022).  

* Rothchild, Daniel, Alex Tamkin, Julie Yu, Ujval Misra, and Joseph Gonzalez. "C5t5: Controllable generation of organic molecules with transformers." arXiv preprint arXiv:2108.10307 (2021).
