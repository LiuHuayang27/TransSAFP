# The data tokenizer for the transfer model. 
# Zilin Song, 2022 OCT 21
# 

import numpy as np
from pretrain_tokenizer import AMPTokenizer, get_csv_dir

if __name__ == "__main__":
    "Run the tokenizer. "
    val_ratio = .2

    tokenizer = AMPTokenizer(15)

    mod_amp_seq, mod_amp_label, mod_amp_nterm = tokenizer.tokenize(get_csv_dir('modified'), ',', 2, col_label=-1, col_nterm=1)

    np.save('./b_preprocessing/transfer/all_seq.npy',   mod_amp_seq)
    np.save('./b_preprocessing/transfer/all_label.npy', mod_amp_label)
    np.save('./b_preprocessing/transfer/all_nterm.npy', mod_amp_nterm)

    print(mod_amp_seq.shape)

    seq_val   = []
    label_val = []
    nterm_val = []
    seq_trn   = []
    label_trn = []
    nterm_trn = []
    
    # For each nterm as i. 
    for i in range(1, 12):

        # Pick out the seqs and corresp. labels with the same nterm.
        i_indices = np.squeeze(mod_amp_nterm==i)

        i_seq   = mod_amp_seq  [i_indices, :]
        i_label = mod_amp_label[i_indices]
        i_nterm = mod_amp_nterm[i_indices]

        # Stat. the labels.
        count     = i_label.shape[0]
        pos_count = i_label[i_label==1].shape[0]
        neg_count = i_label[i_label==0].shape[0]
        
        print(pos_count, neg_count)
        
        # For positive labels. 
        # Stat. number of labels in training and validation
        pos_val_count = int(np.max([1, pos_count*val_ratio]))
        pos_trn_count = pos_count - pos_val_count

        # Pick the positive labels. 
        pos_i_seq   = i_seq  [i_label==1] 
        pos_i_label = i_label[i_label==1] 
        pos_i_nterm = i_nterm[i_label==1]

        # Pick the training positive labels.
        pos_train_indices = np.random.choice(pos_i_seq.shape[0], size=pos_trn_count, replace=False)
        pos_train_i_seq   = pos_i_seq  [pos_train_indices, :]
        pos_train_i_label = pos_i_label[pos_train_indices]
        pos_train_i_nterm = pos_i_nterm[pos_train_indices]
        seq_trn.append  (pos_train_i_seq)
        label_trn.append(pos_train_i_label)
        nterm_trn.append(pos_train_i_nterm)

        # Pick the validations positive labels.
        pos_valid_i_seq   = np.delete(pos_i_seq  , pos_train_indices, axis=0)
        pos_valid_i_label = np.delete(pos_i_label, pos_train_indices, axis=0)
        pos_valid_i_nterm = np.delete(pos_i_nterm, pos_train_indices, axis=0)
        seq_val.append  (pos_valid_i_seq)
        label_val.append(pos_valid_i_label)
        nterm_val.append(pos_valid_i_nterm)

        # For negative labels.
        neg_val_count = int(np.max([1, neg_count*val_ratio]))
        neg_trn_count = neg_count - neg_val_count

        # Pick the negative labels.
        neg_i_seq   = i_seq  [i_label==0]
        neg_i_label = i_label[i_label==0]
        neg_i_nterm = i_nterm[i_label==0]

        # Pick the training negative labels.
        neg_train_indices = np.random.choice(neg_i_seq.shape[0], size=neg_trn_count, replace=False)
        neg_train_i_seq   = neg_i_seq  [neg_train_indices, :]
        neg_train_i_label = neg_i_label[neg_train_indices]
        neg_train_i_nterm = neg_i_nterm[neg_train_indices]
        seq_trn.append  (neg_train_i_seq)
        label_trn.append(neg_train_i_label)
        nterm_trn.append(neg_train_i_nterm)

        neg_valid_i_seq   = np.delete(neg_i_seq,   neg_train_indices, axis=0)
        neg_valid_i_label = np.delete(neg_i_label, neg_train_indices, axis=0)
        neg_valid_i_nterm = np.delete(neg_i_nterm, neg_train_indices, axis=0)
        seq_val.append  (neg_valid_i_seq)
        label_val.append(neg_valid_i_label)
        nterm_val.append(neg_valid_i_nterm)
    
    # List to array. 
    seq_val_array   = seq_val  [0]
    label_val_array = label_val[0]
    nterm_val_array = nterm_val[0]
    seq_trn_array   = seq_trn  [0]
    label_trn_array = label_trn[0]
    nterm_trn_array = nterm_trn[0]

    for i in range(1, len(seq_trn)):
        seq_val_array   = np.concatenate((seq_val_array,   seq_val  [i]), axis=0)
        label_val_array = np.concatenate((label_val_array, label_val[i]), axis=0)
        nterm_val_array = np.concatenate((nterm_val_array, nterm_val[i]), axis=0)
        seq_trn_array   = np.concatenate((seq_trn_array,   seq_trn  [i]), axis=0)
        label_trn_array = np.concatenate((label_trn_array, label_trn[i]), axis=0)
        nterm_trn_array = np.concatenate((nterm_trn_array, nterm_trn[i]), axis=0)

    seq_trn_array   = np.squeeze(seq_trn_array)
    label_trn_array = np.squeeze(label_trn_array)
    nterm_trn_array = np.squeeze(nterm_trn_array)
    seq_val_array   = np.squeeze(seq_val_array)
    label_val_array = np.squeeze(label_val_array)
    nterm_val_array = np.squeeze(nterm_val_array)

    # Stats and save
    print(  seq_trn_array.shape)
    print(label_trn_array.shape)
    print(nterm_trn_array.shape)
    print(  seq_val_array.shape)
    print(label_val_array.shape)
    print(nterm_val_array.shape)

    np.save('./b_preprocessing/transfer/train_seq.npy',   seq_trn_array)
    np.save('./b_preprocessing/transfer/train_label.npy', label_trn_array)
    np.save('./b_preprocessing/transfer/train_nterm.npy', nterm_trn_array)
    np.save('./b_preprocessing/transfer/valid_seq.npy',   seq_val_array)
    np.save('./b_preprocessing/transfer/valid_label.npy', label_val_array)
    np.save('./b_preprocessing/transfer/valid_nterm.npy', nterm_val_array)