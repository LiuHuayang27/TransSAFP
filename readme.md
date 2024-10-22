# TransSAFP

## About TransSAFP
**TransSAFP** is a self-attention based neural network that predicts antimicrobial label of biomimetic active peptides. 

## Environments
The TransSAFP is based on TensorFlow 2.10 (with CudaToolkit 11.2). The bash commands below create a Conda environment for running TransSAFP inference. 
```bash
conda create --name transSAFP
conda activate transSAFP
CONDA_OVERRIDE_CUDA="11.2" conda install tensorflow-gpu=2.10 cudatoolkit==11.2 -c conda-forge
```

## Usage
To install, simply `git clone` this repository and activate the corresponding Conda environment.

To predict a native peptide sequence (without any unnatrual amino acid or chemical modifications) using the pretrain model:
```bash
python run_pretrain.py {Your-Sequence-Here}
python run_pretrain.py AAAAAAAA # Predicts the antimicrobial activity label of the octa-alanine.
```

To predict a chemical modified peptide sequence using the transSAFP model:
```bash
python run_transSAFP.py {Your-Sequence-Here} {Your-Modification-Here}
python run_transSAFP.py AAAAAAAA C-HEX # Predicts the antimicrobial activity label of the octa-alanine with C-HEX N-terminal modification.
```
Available N-terminal types are: 
```
- C8   
- C12  
- C16  
- PHE  
- BIP  
- DIP  
- NAP  
- ANT  
- PYR  
- C-PRO
- C-HEX
```

Prediction scores of SAFPs synthesized in this study is concluded in the file `prediction.log`.

## Correspondence
[Prof. Huaimin Wang](http://www.hm-wanglab.com/) (wanghuaimin@westlake.edu.cn)  
[Prof. Jing Huang](https://github.com/JingHuangLab) (huangjing@westlake.edu.cn)
