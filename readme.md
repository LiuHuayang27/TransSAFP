# TransSAFP

## About TransSAFP
**TransSAFP** is a self-attention based neural network that predicts antimicrobial label of biomimetic active peptides. 

## Environments
The TransSAFP is based on TensorFlow 2.10 (with CudaToolkit 11.2). The terminal commands below create a Conda environment for running TransSAFP inference. 
```bash
conda create --name transSAFP
conda activate transSAFP
CONDA_OVERRIDE_CUDA="11.2" conda install tensorflow-gpu=2.10 cudatoolkit==11.2 -c conda-forge
```

## Usage
To predict a native peptide sequence without any (unnatrual amino acid or chemical modifications), use the pretrain model:
```bash
python run_pretrain.py {Your-Sequence-Here}
python run_pretrain.py AAAAAAAA # Predicts the antimicrobial activity label of octa-alanine.
```

To predict a chemical modified peptide sequence, use the transSAFP model:
```bash
python run_transSAFP.py {Your-Sequence-Here} {Your-Modification-Here}
python run_transSAFP.py AAAAAAAA C6ring # Predicts the antimicrobial activity label of octa-alanine with C6ring modification.
```
## Correspondence
Prof. Huaimin Wang (wanghuaimin@westlake.edu.cn)  
[Prof. Jing Huang](https://github.com/JingHuangLab) (huangjing@westlake.edu.cn)
