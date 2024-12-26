# Mapping all 8-mer peptides to the AMP Transformer for reactivity prediction.
# Zilin Song, 13 Dec 2022
# 

from pretrain_model import AMPPretrainLearner
from transfer_model import AMPTransferLearner
from transfer_model import get_aa_embeding_dists

from itertools import product
import numpy as np
import sys

def load_model():
    # Load the model.
    pretrain_model = AMPPretrainLearner()
    pretrain_model.load_weights('./c_transfer_learning/models_pretrain/amp_pretrain_learner_weights.h5')
    
    pretrain_embedding = pretrain_model.l_we.embedding.embeddings.numpy()
    aa_dists = get_aa_embeding_dists(pretrain_embedding)
    
    transfer_model = AMPTransferLearner(pretrain_model, aa_dists)
    transfer_model.load_weights('./c_transfer_learning/models_transfer/amp_transfer_learner_weights.h5')
    transfer_model.summary()

    return transfer_model

def get_pred(model, seq_1_to_8):
    """Make the prediction with sequences only 1 to 8."""
    # 0. Slices.
    num_seqs  = seq_1_to_8.shape[0]
    per_slice = 3000

    num_slices = num_seqs // per_slice 
    num_slices = num_slices + 1 if num_seqs % per_slice != 0 else num_slices

    x_seq_pos  = None
    x_nter_pos = None
    y_pos      = None

    for i_slice in range(num_slices):
        
        if i_slice != num_slices - 1:   # Regular slice, take per_slice.
            seq_slice = seq_1_to_8[i_slice*per_slice:(i_slice+1)*per_slice]
        else:                           # Last slice, take all that left.
            seq_slice = seq_1_to_8[i_slice*per_slice:                     ]

        # Repeat 11 times for 11 different N-terminis.
        x_seq_slice = np.repeat(seq_slice, repeats=11, axis=0)
        # Append zeros to make 15 aa-length
        x_seq_zeropadding = np.zeros((x_seq_slice.shape[0], 15-seq_slice.shape[1]), dtype=np.int8)
        x_seq_slice = np.concatenate((x_seq_slice, x_seq_zeropadding), axis=1)

        # Make 11-terminis periodically.
        x_nter_slice = np.arange(1, 12, dtype=np.int8)[:, np.newaxis]
        x_nter_slice = np.repeat(x_nter_slice, repeats=seq_slice.shape[0], axis=1)
        x_nter_slice = x_nter_slice.T.flatten()[:, np.newaxis]

        # Make prediction.
        y_slice = model.predict((x_seq_slice, x_nter_slice), batch_size = int(x_seq_slice.shape[0]/20), verbose=0)

        # If there are positive predictions.
        if (np.where(y_slice>=.5)[0]).any():
            
            # If this is the first prediction. 
            if x_seq_pos is None or x_nter_pos is None or y_pos is None:
                x_seq_pos  = x_seq_slice [np.where(y_slice>=.5)[0]]
                x_nter_pos = x_nter_slice[np.where(y_slice>=.5)[0]]
                y_pos      = y_slice     [np.where(y_slice>=.5)[0]]
            
            else:
                x_seq_pos  = np.concatenate((x_seq_pos,  x_seq_slice [np.where(y_slice>=.5)[0]]), axis=0)
                x_nter_pos = np.concatenate((x_nter_pos, x_nter_slice[np.where(y_slice>=.5)[0]]), axis=0)
                y_pos      = np.concatenate((y_pos,      y_slice     [np.where(y_slice>=.5)[0]]), axis=0)
        
        if i_slice % 5 == 0:
            print(f"{seq_1_to_8[0, 0:4]}: "
                  f"{i_slice+1:>4}/{num_slices:<4}; ",
                  flush=True)

    if x_seq_pos is None:
        return None
    else:
        return np.asarray(x_seq_pos, dtype=np.int8), np.asarray(x_nter_pos, dtype=np.int8), np.asarray(y_pos, dtype=np.float64)

if __name__ == '__main__':

    mod = load_model()

    sim_ind = int(sys.argv[1]) # args = [0, 400)

    aa_1 = int(sim_ind // 20 + 1)
    aa_2 = int(sim_ind  % 20 + 1)
    

    all_x_seq_pos  = None
    all_x_nter_pos = None
    all_y_pos      = None

    part_x = 0

    for loop_ind in range(400):

        aa_3 = int(loop_ind // 20 + 1)
        aa_4 = int(loop_ind  % 20 + 1)

        seq_5_to_8_batch = []
        
        for seq in product(np.arange(1, 21), repeat=4):
            seq_5_to_8_batch.append(seq)
        
        seq_1_to_4 = np.asarray([[aa_1, aa_2, aa_3, aa_4]], dtype=np.int8)
        seq_5_to_8 = np.asarray(seq_5_to_8_batch, dtype=np.int8)

        # TODO: in get_pred: 
        #       1. make the actural sequence;
        #       2. return the prediction. 
        seq_1_to_4 = np.repeat(seq_1_to_4, repeats=seq_5_to_8.shape[0], axis=0)
        seq_1_to_8 = np.concatenate((seq_1_to_4, seq_5_to_8), axis=1) # 8 peptide sequences. 
        
        results = get_pred(mod, seq_1_to_8)

        if not results is None:
            seq_pos, nter_pos, y_pos = results[0], results[1], results[2]

            if all_x_seq_pos is None or all_x_nter_pos is None or all_y_pos is None:
                all_x_seq_pos  = seq_pos.copy()
                all_x_nter_pos = nter_pos.copy()
                all_y_pos      = y_pos.copy()
            
            else:
                all_x_seq_pos  = np.concatenate((all_x_seq_pos,  seq_pos ), axis=0)
                all_x_nter_pos = np.concatenate((all_x_nter_pos, nter_pos), axis=0)
                all_y_pos      = np.concatenate((all_y_pos,      y_pos   ), axis=0)
            
            print(f"~~Done {seq_1_to_8[0, 0:4]}: "
                  f"n_pos-{all_y_pos.shape[0]}; "
                  f"  seq_shape-{ all_x_seq_pos.shape}; "
                  f" nter_shape-{all_x_nter_pos.shape}; "
                  f"label_shape-{     all_y_pos.shape}; ", 
                  flush=True)

        else:
            print(f"~~Done {seq_1_to_8[0, 0:4]}: "
                  f"n_pos-0; "
                  f"  seq_shape-0; "
                  f" nter_shape-0; "
                  f"label_shape-0; ", 
                  flush=True)
                  
        if all_x_seq_pos.shape[0] >= 10_000_000:
            np.savez(f'./f_map_seq8_raw/{aa_1:0>2}_{aa_2:0>2}_pt{part_x:0>3}.npz', seq=all_x_seq_pos, nter=all_x_nter_pos, label=all_y_pos)
            part_x += 1
            all_x_seq_pos, all_x_nter_pos, all_y_pos = None, None, None
    
    if not (all_x_seq_pos is None or all_x_nter_pos is None or all_y_pos is None):
      np.savez(f'./f_map_seq8_raw/{aa_1:0>2}_{aa_2:0>2}_pt{part_x:0>3}.npz', seq=all_x_seq_pos, nter=all_x_nter_pos, label=all_y_pos)
