# Evalutations of the transfer model.
# Zilin Song, 2022 Nov 05
# 

import numpy      as np

from pretrain_model import AMPPretrainLearner
from transfer_model import AMPTransferLearner
from transfer_model import get_aa_embeding_dists

if __name__ == "__main__":
    # Prepare the transfer datasets (with nterm). 
    ttrain_seq   = np.load('./b_preprocessing/transfer/train_seq.npy'  )
    ttrain_label = np.load('./b_preprocessing/transfer/train_label.npy')[:, np.newaxis]
    ttrain_nterm = np.load('./b_preprocessing/transfer/train_nterm.npy')[:, np.newaxis]
    tvalid_seq   = np.load('./b_preprocessing/transfer/valid_seq.npy'  )
    tvalid_label = np.load('./b_preprocessing/transfer/valid_label.npy')[:, np.newaxis]
    tvalid_nterm = np.load('./b_preprocessing/transfer/valid_nterm.npy')[:, np.newaxis]

    # Load the model.
    pretrain_model = AMPPretrainLearner()
    pretrain_model.load_weights('./c_transfer_learning/models_pretrain/amp_pretrain_learner_weights.h5')
    
    pretrain_embedding = pretrain_model.l_we.embedding.embeddings.numpy()
    aa_dists = get_aa_embeding_dists(pretrain_embedding)
    
    transfer_model = AMPTransferLearner(pretrain_model, aa_dists)
    transfer_model.load_weights('./c_transfer_learning/models_transfer/amp_transfer_learner_weights.h5')
    transfer_model.summary()

    from sklearn.metrics import recall_score, precision_score, precision_recall_curve, roc_curve, f1_score, accuracy_score

    # Training set metrics with no modifications
    y_true = ttrain_label.copy()
    y_pred = transfer_model(x = (ttrain_seq, ttrain_nterm), training=False)
    # print(y_true.shape)
    # print(y_pred.shape)
    print(f'n_samples: {y_pred.shape[0]}')
    y_pred = np.where(y_pred >=.5, 1, 0)

    precision = precision_score(y_true, y_pred)
    recall    = recall_score   (y_true, y_pred)
    accuracy  = accuracy_score (y_true, y_pred)

    print(f"Mod Training:\nprecision: {precision}\nrecall:    {recall}\naccuracy:  {accuracy}\n")

    # Validation set metrics
    y_true = tvalid_label.copy()
    y_pred = transfer_model(x = (tvalid_seq, tvalid_nterm), training=False)
    # print(y_true.shape)
    # print(y_pred.shape)
    print(f'n_samples: {y_pred.shape[0]}')
    y_pred = np.where(y_pred >=.5, 1, 0)

    precision = precision_score(y_true, y_pred)
    recall    = recall_score   (y_true, y_pred)
    accuracy  = accuracy_score (y_true, y_pred)

    print(f"Mod Validation:\nprecision: {precision}\nrecall:    {recall}\naccuracy:  {accuracy}\n")
    
    # You can tokenize your peptides from a file to the format (list) below. 
    peptides = [
        ['MRRRR',   "C3ring",],
        ['MKKKK',   "C3ring",],
        ['AAAAAA',   "C3ring",],
        # ['KKKKKKKKAAKKKKKK', 'C3ring'], # Peptides longer than 15 will raise an error.
        # ['KKKKKKKKK',        'QQ'],     # Illegal nterm specs will also raise an error.  
    ]

    from b_preprocessing.pretrain_tokenizer import AMPTokenizer
    # Transform the text to numbers. 
    tokenizer = AMPTokenizer(max_seq_len=15)

    seq_list  = []
    nterm_list = []
    
    for pep in peptides:
        seq, nterm = pep[0], pep[1]

        seq_numbers   = tokenizer.tokenize_seq(seq)
        nterm_numbers = tokenizer.tokenize_nterm(nterm)

        seq_list.append(seq_numbers)
        nterm_list.append(nterm_numbers)
    
    # This is the formatted input. 
    seqs = np.asarray(seq_list)
    nterms = np.asarray(nterm_list)

    # Be aware: 'amptl' is the transfer learning model. 
    labels = transfer_model((seqs, nterms), training=False)
    for i in list(zip(labels.numpy(), peptides, seqs)):
        print(i[0][0], i[1], i[2])