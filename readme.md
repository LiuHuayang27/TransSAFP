<!-- # TransBMAP

## About TransBMAP
**TransBMAP** is a self-attention based neural network that predicts antimicrobial label of biomimetic active peptides. 

## Environments
The TransBMAP is based on TensorFlow 2.10. The terminal commands below create a Conda environment for running TransBMAP inference with CudaTOolkit 11.2. 
```bash
conda create --name transbmap
conda activate transbmap
CONDA_OVERRIDE_CUDA="11.2" conda install tensorflow-gpu=2.10 cudatoolkit==11.2 -c conda-forge
```

## Usage
To predict a native peptide sequence without any (unnatrual amino acid or chemical modifications), use the pretrain model:
```bash
python run_pretrain.py {Your-Sequence-Here}
python run_pretrain.py AAAAAAAA # Predicts the antimicrobial activity label of octa-alanine.
```

To predict a chemical modified peptide sequence, use the transBMAP model:
```bash
python run_transbmap.py {Your-Sequence-Here} {Your-Modification-Here}
python run_transbmap.py AAAAAAAA C6ring # Predicts the antimicrobial activity label of octa-alanine with C6ring modification.
```
## Correspondence
Prof. Huaimin Wang (wanghuaimin@westlake.edu.cn)  
Prof. Jing Huang (huangjing@westlake.edu.cn) -->