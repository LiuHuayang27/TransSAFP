# The layers for building the pretraining transformer model. 
# Zilin Song, 2022 OCT 10
# 

import tensorflow as tf
import numpy      as np

class AMPPretrainSeqEmbeddingPass(tf.keras.layers.Layer):
    """The pretraining word embedding pass. """

    def __init__(self,
                 vocab_size: int, 
                 d_model:    int):
        """Create a word embedding layer.
        
        Args:
            vocal_size (int): Embedding layer, the size of dictionary (no. aa letters);
            d_model    (int): Embedding layer, the length of the embedding vector for each aa letter;
        """
        super().__init__()
        self._d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_dim  = vocab_size,
                                                   output_dim = d_model,
                                                   mask_zero  = True)
    
    def call(self, x, training: bool = True):
        x  = self.embedding(x, training=training)
        x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32)) # Scale to be compatible to pos encoding
        return x

class AMPPretrainPosEmbeddingPass(tf.keras.layers.Layer):
    """The pretraining positional embedding pass."""
    
    def __init__(self, 
                 max_seq_len: int, 
                 d_model:     int):
        """Create a positional embedding layer.
        
        Args:
            max_seq_len (int): The length of the longest sequence. 
            d_model     (int): The length of the embedding vector for each aa letter;
        """
        super().__init__()
        pe = self.abs_pos_embedding(max_seq_len, d_model)
        self.pos_embed_vec = tf.cast(pe, dtype=tf.float32)[tf.newaxis, :, :]

    def call(self, x, training: bool = None):
        x += self.pos_embed_vec
        return x

    def abs_pos_embedding(self,
                          max_seq_len: int, 
                          d_model: int):
        """Implements the absolute positional embedding of 
        arXiv:1706.03762 at sec. 3.5. 
        Args:
            max_seq_len (int): The length of the longest sequence. 
            d_model     (int): The length of the embedding vector for each aa letter;
        """
        pos   = np.arange(max_seq_len)[:, np.newaxis]
        two_i = np.arange(0, d_model, 2) / d_model
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(pos / (np.power(10000., two_i)))
        pe[:, 1::2] = np.cos(pos / (np.power(10000., two_i)))
        return pe

class ResconSelfAttntPass(tf.keras.layers.Layer):
    """The residual connection self-attention: Input -> MHA -> add&norm -> Output. """

    def __init__(self, 
                 d_model:    int, 
                 mha_nheads: int, 
                 mha_dorate: float):
        """Create a residual connection self-attention pass. 
        
        Args:
            d_model      (int): MHA layer: the length of embedding vector for each aa letter; 
            mha_nheads   (int): The number of self-attention heads in MHA;
            mha_dorate (float): The dropout rate for the MHA layer.
        """
        super().__init__()

        # The MHA pass. 
        self.mhatt = tf.keras.layers.MultiHeadAttention(num_heads = mha_nheads,
                                                        key_dim   = d_model, 
                                                        dropout   = mha_dorate)
        self.mhatt_add   = tf.keras.layers.Add()
        self.mhatt_lnorm = tf.keras.layers.LayerNormalization()

    def call(self, 
             x, 
             training: bool = True):
        # mha -> add&norm
        ## MHA pass. 
        q, v, k = x
        mhatt_out, mhatt_score = self.mhatt(query = q, 
                                            value = v, 
                                            key   = k, 
                                            training=training, 
                                            return_attention_scores = True)
        self.last_attnt_scores = mhatt_score
        x         = self.mhatt_add  ([q, mhatt_out], training=training) # query is added.
        x         = self.mhatt_lnorm(x,              training=training)

        return x

class ResconFeedForwdPass(tf.keras.layers.Layer):
    """The residual connection feed forward: Input -> FFNN -> add&norm -> Output."""

    def __init__(self, 
                 d_model:    int,
                 ff_l1units: int,
                 ff_dorate:  float):
        """Create a residual connected feedforward pass.
        
        Args:
            d_model      (int): FF layer: the length of embedding vector for each aa letter; 
            ff_l1units   (int): The number of units in the first FF/FC layer;
            ff_dorate  (float): The dropout rate after the FF/FC embedding. 
        """
        super().__init__()

        # The FFNN pass. 
        self.ffnn_l1    = tf.keras.layers.Dense(units  = ff_l1units, activation='relu')
        self.ffnn_lfin  = tf.keras.layers.Dense(units  = d_model)
        self.ffnn_drp   = tf.keras.layers.Dropout(rate = ff_dorate)
        self.ffnn_add   = tf.keras.layers.Add()
        self.ffnn_lnorm = tf.keras.layers.LayerNormalization()

    def call(self, 
             x, 
             training: bool = True):
        # FFNN -> add&norm
        ffnn_out = self.ffnn_l1   (x,             training=training)
        ffnn_out = self.ffnn_lfin (ffnn_out,      training=training)
        ffnn_out = self.ffnn_drp  (ffnn_out,      training=training)
        x        = self.ffnn_add  ([x, ffnn_out], training=training)
        x        = self.ffnn_lnorm(x,             training=training)

        return x

class AMPPretrainTransformerPass(tf.keras.layers.Layer):
    """The pretraining transformer model, without input positional embeddings."""
    
    def __init__(self, 
                 max_seqlen: int,
                 d_model:    int, # dim for word embedding.  
                 all_mha_nheads: int   =   8,
                 all_mha_dorate: float =  .2,
                 all_ffn_l1unit: int   = 128,
                 all_ffn_dorate: float =  .2):
        """Create an transformer pretraining learner. 

        Args:
            max_seqlen   (int): the length of the longest sequence (aa sequence);
            d_model      (int): embedding size of each word (each aa letter), from the PosEmbedder;
        """
        super().__init__()

        self.prm_max_seqlen = max_seqlen
        self.prm_d_model    = d_model

        # Input dropout.
        self.inp_dpout = tf.keras.layers.Dropout(rate=all_ffn_dorate)

        # Encoder pass.
        ## The global self attention pass.
        self.enc_global_self_attnt = ResconSelfAttntPass(d_model    = d_model,
                                                         mha_nheads = all_mha_nheads,
                                                         mha_dorate = all_mha_dorate)
        self.enc_global_feedforwdn = ResconFeedForwdPass(d_model    = d_model,
                                                         ff_l1units = all_ffn_l1unit,
                                                         ff_dorate  = all_ffn_dorate)
        
        # Decoder pass.
        ## The causal self attention pass: 
        ## Note that no causal mask was made and the full seq is known.
        ## we aim to rebuild (not translate) the sequences. 
        self.dec_causal_self_attnt = ResconSelfAttntPass(d_model    = d_model,
                                                         mha_nheads = all_mha_nheads,
                                                         mha_dorate = all_mha_dorate)

        ## The cross self attention pass.
        self.dec__cross_self_attnt = ResconSelfAttntPass(d_model    = d_model,
                                                         mha_nheads = all_mha_nheads,
                                                         mha_dorate = all_mha_dorate)
        self.dec__cross_feedforwdn = ResconFeedForwdPass(d_model    = d_model,
                                                         ff_l1units = all_ffn_l1unit,
                                                         ff_dorate  = all_ffn_dorate)

        # Final layer. # Simple flatten (may try Conv1D and Pooling also?)
        self.lout_flatten = tf.keras.layers.Flatten()
        self.lout_label   = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)
        self.lout_seq     = tf.keras.layers.Dense(d_model,                 use_bias=False)

    def call(self, 
             x, 
             training: bool = True):
        """Convinient call for inputs on models. 

        Args:
            x   (tf.tensor):  1-20 encoded seq;
            training (bool):  state = training or not.
        """
        x = self.inp_dpout(x, training=training)

        c = tf.identity(x)

        # Encoder pass. 
        ## Global self attention pass.
        c = self.enc_global_self_attnt((c, c, c), training=training)
        c = self.enc_global_feedforwdn(    c    , training=training)

        # Decoder pass.
        ## Causal self attention pass (with no causal mask).
        x = self.dec_causal_self_attnt((x, x, x), training=training)

        ## Cross self attention pass.
        x = self.dec__cross_self_attnt((x, c, c), training=training)
        ## Record the cross attention scores and latent variable. 
        self.cross_attn_latent = x
        self.cross_attn_scores = self.dec__cross_self_attnt.last_attnt_scores

        x = self.dec__cross_feedforwdn(    x    , training=training)

        # Reconstructed sequences. 
        c_recon = self.lout_seq(x, training=training)

        # final activity output. 
        # print(x.shape). # Modify here for sequence output. 
        x = self.lout_flatten(x, training=training)
        x = self.lout_label  (x, training=training)

        return x, c_recon
