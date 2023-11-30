# The data tokenizer for the pretrain model. 
# Zilin Song, 2022 OCT 21
# 

import numpy as np
    
if __name__ == "__main__":
    seq_trn            = np.load('./b_preprocessing/pretrain/train_seq.npy',   )
    label_trn          = np.load('./b_preprocessing/pretrain/train_label.npy', )
    baseline_seq_tst   = np.load('./b_preprocessing/pretrain/valid_seq.npy',   )
    baseline_label_tst = np.load('./b_preprocessing/pretrain/valid_label.npy', )

    pos_label_trn = np.where(label_trn==1.)[0]
    neg_label_trn = np.where(label_trn==0.)[0]

    baseline_seq_val   = np.concatenate((seq_trn  [pos_label_trn][:int(pos_label_trn.shape[0]/4)], seq_trn  [neg_label_trn][:int(neg_label_trn.shape[0]/4)], ))
    baseline_label_val = np.concatenate((label_trn[pos_label_trn][:int(pos_label_trn.shape[0]/4)], label_trn[neg_label_trn][:int(neg_label_trn.shape[0]/4)], ))
    baseline_seq_trn   = np.concatenate((seq_trn  [pos_label_trn][int(pos_label_trn.shape[0]/4):], seq_trn  [neg_label_trn][int(neg_label_trn.shape[0]/4):], ))
    baseline_label_trn = np.concatenate((label_trn[pos_label_trn][int(pos_label_trn.shape[0]/4):], label_trn[neg_label_trn][int(neg_label_trn.shape[0]/4):], ))

    print(seq_trn.shape)
    print(baseline_seq_trn.shape)
    print(baseline_label_trn)
    print(baseline_seq_val.shape)
    print(baseline_label_val)
    print(baseline_seq_tst.shape)
    print(baseline_label_tst)

    np.save('./b_preprocessing/pretrain/baseline_train_seq'  , baseline_seq_trn  )
    np.save('./b_preprocessing/pretrain/baseline_train_label', baseline_label_trn)
    np.save('./b_preprocessing/pretrain/baseline_valid_seq'  , baseline_seq_val  )
    np.save('./b_preprocessing/pretrain/baseline_valid_label', baseline_label_val)
    np.save('./b_preprocessing/pretrain/baseline_test_seq'  ,  baseline_seq_tst  )
    np.save('./b_preprocessing/pretrain/baseline_test_label',  baseline_label_tst)