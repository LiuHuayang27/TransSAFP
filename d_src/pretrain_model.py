# Supervised training of pretraining transformer model. 
# Zilin Song, 2022 OCT 11
# 

import tensorflow as tf
from c_transfer_learning.layers.pretrain_layers import (AMPPretrainPosEmbeddingPass, 
                                                        AMPPretrainSeqEmbeddingPass, 
                                                        AMPPretrainTransformerPass , )

class AMPPretrainLearner(tf.keras.Model):
    """The pretraining model. """

    def __init__(self, max_seqlen=15, d_model=24):
        """Create the pretraining model. 
        """
        super().__init__()

        self.prm_max_seqlen = max_seqlen
        self.prm_d_model    = d_model

        # AA letter embedding.
        self.l_we = AMPPretrainSeqEmbeddingPass(vocab_size=21, 
                                                d_model=d_model)

        # Positional encoding.
        self.l_pe = AMPPretrainPosEmbeddingPass(max_seqlen, d_model)

        # Transformer pass.
        self.l_tf = AMPPretrainTransformerPass(max_seqlen, d_model)

        self.build(input_shape=(max_seqlen, ))

        # Metric takers. 
        self.total_loss_tracker = tf.keras.metrics.Mean(name="tot_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.binry_loss_tracker = tf.keras.metrics.Mean(name="bin_loss")

    def call(self, 
             x, 
             training: bool = True):

        x_embed    = self.l_we(x,       training)
        x          = self.l_pe(x_embed, training)
        y, x_recon = self.l_tf(x,       training)

        # For convenient calls.
        self.cross_attn_latent = self.l_tf.cross_attn_latent
        self.cross_attn_scores = self.l_tf.cross_attn_scores

        return y, x_embed, x_recon

    # Custom training step.
    def train_step(self, data):
        """Custom training step for loss customization. 
        Overrides the inherited keras.Model.train_step().
    
        The model has to:
        1. Reconstruct the encoder context embedding. 
        2. Correctly predict the AMP activity label. 
        """
        # Unpack
        x, y_true = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred, x_embed, x_recon = self(x, training=True)

            # Compute losses
            rec_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x_embed, x_recon)) 
            bin_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            tot_loss = rec_loss + bin_loss

        # Backward pass
        grads = tape.gradient(tot_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Record losses
        self.total_loss_tracker.update_state(tot_loss)
        self.recon_loss_tracker.update_state(rec_loss)
        self.binry_loss_tracker.update_state(bin_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self,data):
        """Custom evaluation step. """
        # Unpack data
        x, y_true = data
        
        # Forward pass
        y_pred, x_embed, x_recon = self(x, training=False)

        # Compute losses
        recon_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x_embed, x_recon)) 
        binry_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        total_loss = recon_loss + binry_loss

        # Record losses
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.binry_loss_tracker.update_state(binry_loss)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # This is the self.metrics.
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.binry_loss_tracker]