# TransSAFP

## About TransSAFP
**TransSAFP** is a self-attention based neural network that predicts antimicrobial label of self-assembling functional peptides. 

## Environments
The TransSAFP is based on TensorFlow 2.10 (with CudaToolkit 11.2). The bash commands below create a Conda environment for running TransSAFP inference. 
```bash
conda create --name transSAFP
conda activate transSAFP
CONDA_OVERRIDE_CUDA="11.2" conda install tensorflow-gpu=2.10 cudatoolkit==11.2 -c conda-forge
```

## Usage
To install, simply `git clone` this repository and activate the corresponding Conda environment.

To predict a native peptide sequence (without any unnatural amino acid or chemical modifications) using the pretraining model:
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

## Description for files and directories.
- `./a_raw_datasets`:   Contains the N-terminal modified SAFPs data for the model training;
- `./b_preprocessing`:  Contains the tokenizers;
- `./c_transsafp`:      Contains the model weights;
- `./d_src`:            Contains the source codes for the TransSAFP model:
  - `./d_src/a_raw_datasets`:         Contains the N-terminal modified SAFPs data for the model training;
  - `./d_src/b_preprocessing`:        Contains the tokenizers and the transfer learning model and the preprocessed N-terminal modified SAFPs data;
  - `./d_src/c_transfer_learning`:    Contains the layers implemented for the TransSAFP model and the model weights;
  - `./d_src/map_seq8.py`:            The source codes for mapping the N-terminal octa-peptide space;
  - `./d_src/pretrain_model.py`:      The source codes for the pretraining model;
  - `./d_src/transfer_model.py`:      The source codes for the TransSAFP model;
  - `./d_src/transfer_model_eval.py`: The source codes for evaluating the TransSAFP model;
  - `./d_src/*.ipynb`:                Analysis of the TransSAFP mode latent space;
- `./prediction.log`:   The prediction scores of the 140 SAFPs predicted and experimented synthesized in this study;
- `./run_pretrain.py`:  An interface for single sample inference of the pretraining model;
- `./run_transsafp.py`: An interface for single sample inference of the TransSAFP model.


## References

**"De novo design of self-assembling peptides with antimicrobial activity guided by deep-learning"**  
Huayang Liu†, Zilin Song†, Yu Zhang, Bihan Wu, Dinghao Chen, Ziao Zhou, Hongyue Zhang, Sangshuang Li, Xinping Feng, Jing Huang*, Huaimin Wang*


**Correspondence:**  
[Prof. Huaimin Wang](http://www.hm-wanglab.com/) (wanghuaimin@westlake.edu.cn)  
[Prof. Jing Huang](https://github.com/JingHuangLab) (huangjing@westlake.edu.cn)