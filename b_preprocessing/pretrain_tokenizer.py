# The data tokenizer for the pretrain model. 
# Zilin Song, 2022 OCT 21
# 

import numpy as np

class AMPTokenizer(object):
    """Implements a tokenizer for peptide seq tokenization. """

    # 20 amino acids
    aa_dict = {"A": 1, 
               "C": 2, 
               "D": 3, 
               "E": 4, 
               "F": 5,
               "G": 6, 
               "H": 7, 
               "I": 8, 
               "K": 9, 
               "L": 10,
               "M": 11,
               "N": 12,
               "P": 13,
               "Q": 14,
               "R": 15,
               "S": 16,
               "T": 17,
               "V": 18,
               "W": 19,
               "Y": 20,
               # "X": 0, Removed all sequences that contains an 'X'.
    }

    # N-terminals
    nterm_dict = {"C8":     1,
                  "C12":    2,
                  "C16":    3,
                  "1P":     4,
                  "BIP":    5,
                  "DIP":    6,
                  "NAL":    7,
                  "ATH":    8,
                  "PYR":    9,
                  "C3ring": 10,
                  "C6ring": 11,
    }

    def __init__(self, 
                 max_seq_len: int):
        """Create a tokenizer object. 
        
        Args:
            max_seq_len (int): Denotes the maximum length of the aa sequence. 
        """
        self._max_seq_len = max_seq_len

    def tokenize_seq(self, seq: str):
        """Tokenize one aa sequence. 
        
        Args:
            seq (str): the string of single letter amino acid sequences.
        """
        tokens = np.zeros((self._max_seq_len, ))
        
        for i_aa in range(len(seq)):
            tokens[i_aa] = self.aa_dict[seq[i_aa]]
            
        return tokens

    def tokenize_nterm(self, nterm: str):
        """Tokenize one nterm label. """
        tokens = np.zeros((1, ))
        tokens[0] = self.nterm_dict[nterm]

        return tokens

    def tokenize_label(self, label: str):
        """Tokenize one activity label. """
        tokens = int(label)
        return tokens

    def tokenize(self, 
                 file_dir:  str, 
                 splitter:  str, 
                 col_seq:   int, 
                 col_label: int = None, 
                 col_nterm: int = None, 
                 skip:      int = 1):
        """Tokenize one line in csv file. 
        
        Args:
            file_dir  (str): the directory of the csv file. 
            splitter  (str): the string to split lines into fields.
            col_seq   (int): the column index of the sequence. 
            col_label (int): the column index of the label. 
            col_nterm (int): the column index of the nterm. 
            skip      (int): the number of lines to skip. 
        """
        seq_list   = []
        label_list = []
        nterm_list = []
        
        with open(file_dir, 'r') as reader:
            lines = reader.readlines()

            for i_line in range(skip, len(lines)):
                line = lines[i_line]

                if 'X' in line: # Note that all lines contain 'X' were ignored. 
                    continue
                if ',,,,' == line[:4]:
                    continue

                words = line.split(splitter)

                seq   = words[col_seq  ].strip()
                label = words[col_label].strip() if not col_label is None else None
                nterm = words[col_nterm].strip() if not col_nterm is None else None
                
                # Removed duplicated entries in 62000.csv
                # KYFLGKKRII
                # FKRFAKKF
                # 
                # if not seq in seq_list:
                #     seq_list.append(seq)
                # else:
                #     print(seq)
                #     continue
                seq_list.append(seq)

                if not label is None:
                    label_list.append(label)
                
                if not nterm is None:
                    nterm_list.append(nterm)

        seq_array   = []
        for seq in seq_list:
            seq_array.append(self.tokenize_seq(seq))
        
        label_array = []
        for label in label_list:
            label_array.append(self.tokenize_label(label))

        nterm_array = []
        for nterm in nterm_list:
            nterm_array.append(self.tokenize_nterm(nterm))
        
        return np.asarray(seq_array), np.asarray(label_array), np.asarray(nterm_array)

def get_csv_dir(which_csv: str):         
    """This function returns the directory string of the csv files. 
    which_csv: unmodified_pos / unmodified_neg / modified
    """ 
    # """xxx""" is a docstring for notating def functions

    basedir = './a_raw_datasets' # Base directory of the data sets. 

    if   which_csv == 'unmodified_pos':                     # The directory to the unmodified AMP csv. 
        targetdir = f'{basedir}/noN_AMP_100ugml_3200.csv'   # string led by an 'f' means that contents in {} should be the value of the variables.
    
    elif which_csv == 'unmodified_neg':                     # The directory to the unmodified non-AMP csv. 
        targetdir = f'{basedir}/noN_Non_AMPs_62000.csv'
    
    elif which_csv == 'modified':                           # The directory to the modified AMP/non-AMP csv. 
        targetdir = f'{basedir}/Nmodified_peptides.csv'
    
    else:                                                   # Passing the wrong kw for which_csv raises an error. 
        raise Exception('Wrong specs of which_csv!')

    return targetdir

def mk_pretraining_sets(x, y, valid_split=.2):
    """Stratified split for training, validation sets. 
    It means that the validation and training sets has the same ratio of positive / negative data set. 
    So that the positive ~ negative distribution of data in the training set is the same to the validation set. 
    """
    x_neg = x[np.where(y==0)[0]]  # Find non-active sequences. 
    y_neg = y[np.where(y==0)[0]]  # Find non-active sequence labels. 
    
    neg_val_count = int(float(y_neg.shape[0]) * valid_split)                                 # Total number of non-actice set. 
    neg_val_indices = np.random.choice(y_neg.shape[0], size=neg_val_count, replace=False)    # Randomly pick the validation indices. 

    x_neg_val = x_neg[neg_val_indices, :]   # The negative set in the validation set. 
    y_neg_val = y_neg[neg_val_indices]      # The negative  labels in the validation set.
    x_neg_trn = np.delete(x_neg, neg_val_indices, axis=0)    # The negative set in the training set.
    y_neg_trn = np.delete(y_neg, neg_val_indices, axis=0)    # The negative  labels in the training set. 
    
    # Below repeat the same but picks the positive data. 
    x_pos = x[np.where(y==1)[0]]  # Find active sequences.
    y_pos = y[np.where(y==1)[0]]  # Find active sequence labels

    pos_val_count = int(float(y_pos.shape[0]) * valid_split)
    pos_val_indices = np.random.choice(y_pos.shape[0], size=pos_val_count, replace=False)
    
    x_pos_val = x_pos[pos_val_indices, :]
    y_pos_val = y_pos[pos_val_indices]
    x_pos_trn = np.delete(x_pos, pos_val_indices, axis=0)
    y_pos_trn = np.delete(y_pos, pos_val_indices, axis=0)

    print(f'Pretraining set...\n\n'
          f'Total set    {x.shape[0]}-{y.shape[0]},         x.shape = {x.shape},         y.shape = {y.shape}        \n'
          f'Positive set {x_pos.shape[0]}-{y_pos.shape[0]}, x_pos.shape = {x_pos.shape}, y_pos.shape = {y_pos.shape}\n'
          f' Training   x_pos_trn = {x_pos_trn.shape}, y_pos_trn = {y_pos_trn.shape}                                \n'
          f' Validation x_pos_val = {x_pos_val.shape}, y_pos_val = {y_pos_val.shape}                                \n'
          f'Negative set {x_neg.shape[0]}-{y_neg.shape[0]}, x_neg.shape = {x_neg.shape}, y_neg.shape = {y_neg.shape}\n'
          f' Training   x_neg_trn = {x_neg_trn.shape}, y_neg_trn = {y_neg_trn.shape}                                \n'
          f' Validation x_neg_val = {x_neg_val.shape}, y_neg_val = {y_neg_val.shape}                                \n'
    )

    # Concatenate the training and validation sets. 
    # training sets.
    x_trn = np.concatenate((x_pos_trn, x_neg_trn), axis=0)
    y_trn = np.concatenate((y_pos_trn, y_neg_trn), axis=0)

    # validation sets. 
    x_val = np.concatenate((x_pos_val, x_neg_val), axis=0)
    y_val = np.concatenate((y_pos_val, y_neg_val), axis=0)

    return x_trn, y_trn, x_val, y_val
    
if __name__ == "__main__":
    "Run the tokenizer. "
    tokenizer = AMPTokenizer(15)

    # Pretraining sets. 
    raw_true_amp_seq, _, _  = tokenizer.tokenize(get_csv_dir('unmodified_pos'), ',', 2)
    raw_true_amp_label      = np.ones((raw_true_amp_seq.shape[0], ))
    raw_false_amp_seq, _, _ = tokenizer.tokenize(get_csv_dir('unmodified_neg'), ',', 2)
    raw_false_amp_label     = np.zeros((raw_false_amp_seq.shape[0], ))
    raw_amp_seq   = np.concatenate((raw_true_amp_seq,   raw_false_amp_seq  ), axis=0)
    raw_amp_label = np.concatenate((raw_true_amp_label, raw_false_amp_label), axis=0)
        
    print(f'Shapes of sequences and labels.                                    \n'
          f'Unmodified AMP-positive   seq encoding: {raw_true_amp_seq.shape   }\n'
          f'Unmodified AMP-positive label encoding: {raw_true_amp_label.shape }\n'
          f'Unmodified AMP-negative   seq encoding: {raw_false_amp_seq.shape  }\n'
          f'Unmodified AMP-negative label encoding: {raw_false_amp_label.shape}\n'
          f'Unmodified merged         seq encoding: {raw_amp_seq.shape        }\n'
          f'Unmodified merged       label encoding: {raw_amp_label.shape      }\n')

    np.save('./b_preprocessing/pretrain/all_seq.npy',   raw_amp_seq  )
    np.save('./b_preprocessing/pretrain/all_label.npy', raw_amp_label)

    # Split pretraining data to training/validation.
    seq_trn, label_trn, seq_val, label_val = mk_pretraining_sets(raw_amp_seq, raw_amp_label)

    np.save('./b_preprocessing/pretrain/train_seq.npy',   seq_trn  )
    np.save('./b_preprocessing/pretrain/train_label.npy', label_trn)
    np.save('./b_preprocessing/pretrain/valid_seq.npy',   seq_val  )
    np.save('./b_preprocessing/pretrain/valid_label.npy', label_val)
