# The layers for building the transfer learning model. 
# Zilin Song, 2022 OCT 22
# 

import tensorflow as tf
import numpy      as np

from keras                   import backend
from keras.utils             import tf_utils
from keras.engine.base_layer import BaseRandomLayer

class AMPUniformNoisePass(BaseRandomLayer):
    """Layer to add random noise to the AA residues. 

    Modified from TensorFlow source code - tf.keras.layers.GaussianNoise.
    https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GaussianNoise
    """

    def __init__(self, 
                 aa_dists, 
                 seed=None, 
                 **kwargs): 
        """Create a layer to add random noise to each AA letter from a uniform distribution.
        
        Args:
            aa_dists        : the distances between one AA letter to other AA letters; 
            seed            : the random seed.
        """
        super().__init__(seed=seed, **kwargs)
        
        # minimum radius for each AA (include 0).
        self.aa_min_dists = tf.divide(tf.constant(np.min(aa_dists, axis=1)), 2.)

        self.supports_masking = True
        self.seed             = seed

    def call(self, x, training=None):
        """Inputs.shape = ((seq_len, d_embed), (20,))"""

        def noised():
            # Unpack inputs
            seq_embed, seq_index = x
            # cast to int. 
            seq_index = tf.cast(seq_index, dtype=tf.int32)

            # Get the min distance at each aa letter in the sample sequences. 
            seq_aa_min_dists = tf.nn.embedding_lookup(self.aa_min_dists, seq_index)
            # Randomly sample distances in the range of [0, 1] for each aa letter. 
            seq_aa_rnd_dists = self._random_generator.random_uniform(shape  = tf.shape(seq_aa_min_dists),
                                                                     minval = 0., maxval = 1.,
                                                                     dtype  = seq_embed.dtype)
            # Scale the random distances to the range of [0, min_dist].
            seq_aa_rnd_dists = tf.multiply(seq_aa_rnd_dists, seq_aa_min_dists)

            # Randomly sample the coordinates on the embedded aa sequences. 
            seq_rnd_embed = self._random_generator.random_uniform(shape  = tf.shape(seq_embed),
                                                                  minval = -.5, maxval =.5,
                                                                  dtype  = seq_embed.dtype)
            # Get the scaling factor between the 2-norm rnd_embed and the rnd_dists
            # Then scale rnd_embed to get the noise such that the distance between
            # the noised point and the raw point is equal to seq_aa_rnd_dists. 
            seq_rnd_embed_scalfac = seq_aa_rnd_dists / tf.norm(seq_rnd_embed, ord=2, axis=-1)
            seq_noise             = seq_rnd_embed * tf.expand_dims(seq_rnd_embed_scalfac, axis=-1)

            # # For debug.
            # print('~~~~~')
            # print(self.aa_min_dists)
            # print(seq_index)
            # print(seq_aa_min_dists>tf.norm(seq_noise, ord=2, axis=-1))
            # print(tf.norm(seq_noise,     ord=2, axis=-1))
            # print(tf.norm(seq_rnd_embed, ord=2, axis=-1))
            # print('~~~~~')

            return seq_embed + seq_noise

        return backend.in_train_phase(noised, x[0], training=training)

    def get_config(self):
        config = {"seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
