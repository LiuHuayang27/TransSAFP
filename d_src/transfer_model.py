# Supervised training of transfer model.
# Zilin Song 2022 OCT 26
# 

import tensorflow as tf
import numpy      as np

from pretrain_model import AMPPretrainLearner
from c_transfer_learning.layers.transfer_layers import AMPUniformNoisePass

class AMPTransferLearner(tf.keras.Model):
    """The transfer learning model. """

    def __init__(self,
                 pretrain_model:    AMPPretrainLearner,
                 aa_dists:          np.ndarray, 
                 nterm_vocab_size:  int = 12,
                 all_dorate:        float = .2):
        """Create the transfer learning model. """
        super().__init__()

        # The pretraining model segments. 
        self.pretrain_l_we = pretrain_model.l_we
        self.pretrain_l_pe = pretrain_model.l_pe
        # Pretraining model segments. 
        self.pretrain_l_tf = pretrain_model.l_tf

        self.pretrain_l_we.trainable = False
        self.pretrain_l_tf.trainable = False
        
        # The noise augmentation on aa letter embeddings. 
        self.aug_noise_pass = AMPUniformNoisePass(aa_dists)
        
        # Prepend nterm embeddings. 
        self.nterm_embed = tf.keras.layers.Embedding(input_dim=nterm_vocab_size, 
                                                     output_dim=pretrain_model.prm_d_model, 
                                                     mask_zero=True)
        self.nterm_reshp = tf.keras.layers.Reshape(target_shape=(1, pretrain_model.prm_d_model))
        self.concat_nterm_seq = tf.keras.layers.Concatenate(axis=1)

        # The dropout augmentation on sequence embeddings. 
        self.aug_dpout_pass = tf.keras.layers.Dropout(rate=all_dorate)

        # ATTENT pass.
        self.attnt_mha = tf.keras.layers.MultiHeadAttention(num_heads = 8,
                                                            key_dim   = pretrain_model.prm_d_model, 
                                                            dropout   = all_dorate)
        self.attnt_add = tf.keras.layers.Add()
        self.attnt_lyn = tf.keras.layers.LayerNormalization()
        
        # The output pass
        self.lout_flatten = tf.keras.layers.Flatten()
        self.lout_denserl = tf.keras.layers.Dense(units=16, activation='relu')

        self.lout_drpout = tf.keras.layers.Dropout(all_dorate)
        self.lout_label  = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=False)

        self.build(input_shape=[(pretrain_model.prm_max_seqlen, ), (1, )])

    def call(self, x, training: bool = None):
        
        # Unpack inputs. 
        seq_aa, nterm = x

        # ==== Seq pretraining pass ====

        # Seq embedding: pretrain word embedding.
        seq = self.pretrain_l_we(seq_aa, training=False)

        # Seq augmentation: random noise pass.
        seq = self.aug_noise_pass((seq, seq_aa), training=training)

        # Seq embedding: pretrain positional embedding.
        seq = self.pretrain_l_pe(seq, training=False)

        # Seq pretraining transformer pass.
        seq_x = self.pretrain_l_tf.inp_dpout(seq, training=False)
        seq_c = tf.identity(seq_x)

        # Pretrain transformer encoder global self attention.
        seq_c = self.pretrain_l_tf.enc_global_self_attnt((seq_c, seq_c, seq_c), training=False)
        seq_c = self.pretrain_l_tf.enc_global_feedforwdn( seq_c,                training=False)

        # Pretrain transformer decoder causal self attention. 
        seq_x = self.pretrain_l_tf.dec_causal_self_attnt((seq_x, seq_x, seq_x), training=False)

        # Pretrain transformer decoder  cross self attention.
        seq = self.pretrain_l_tf.dec__cross_self_attnt((seq_x, seq_c, seq_c), training=False)
        seq = self.pretrain_l_tf.dec__cross_feedforwdn( seq                 , training=False) 

        # ========= NTerm pass =========

        # Nterm embedding pass.
        nterm = self.nterm_embed(nterm, training=training)
        nterm = self.nterm_reshp(nterm, training=training)

        # Concat.
        x = self.concat_nterm_seq([nterm, seq], training=training)

        # Seg augmentation: dropout pass
        x = self.aug_dpout_pass(x, training=training)

        attnt_out, attnt_scores = self.attnt_mha(query = x,
                                                 value = x,
                                                 key   = x,
                                                 training = training,
                                                 return_attention_scores = True)
        self.last_attention_scores = attnt_scores

        x = self.attnt_add([attnt_out, x], training=training)
        x = self.attnt_lyn(x,              training=training)

        # Output
        x = self.lout_flatten(x, training=training)
        x = self.lout_denserl(x, training=training)

        self.latent_vec = x

        x = self.lout_drpout (x, training=training)
        x = self.lout_label  (x, training=training)

        return x
    
def get_sample_weight(nterm: np.ndarray):
    """Get sample weights.
    nterm includes pretrain labels (which are zeros).
    """
    cls_ind = []
    
    for i in range(12):
        ind = np.where(nterm==i)[0]
        cls_ind.append(ind)

    weights = np.zeros(nterm.shape)

    for i in range(12):
        if cls_ind[i].shape[0] != 0:
            weights[cls_ind[i]] = nterm.shape[0] / 12. / cls_ind[i].shape[0]
        else:
            weights[cls_ind[i]] = 0.

    return weights

def get_aa_embeding_dists(seq_embeddings: np.ndarray):
    """Get the 2-norm distances between aa embeddings. """

    all_dists = []

    for i in range(0, seq_embeddings.shape[0]):
        dists = []

        for j in range(0, seq_embeddings.shape[0]):
            if i == j: continue

            aa_i = seq_embeddings[i]
            aa_j = seq_embeddings[j]

            dist = np.linalg.norm(np.abs(np.asarray(aa_i - aa_j)))

            dists.append(dist)
        
        all_dists.append(dists)

    return all_dists