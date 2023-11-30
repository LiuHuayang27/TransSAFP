# Run the pretraining module of TransBMAP.
# Zilin Song 20221011
# 

import sys
from b_preprocessing.pretrain_tokenizer import AMPTokenizer

if __name__ == "__main__":
  # Sanity checks. 
  assert len(    sys.argv    ) == 2,  "run_pretrain.py takes only the sequences."

  sequence = str(sys.argv[1])

  assert len(sequence) <= 15, "run_pretrain.py takes only sequences with <= 15 AAs."
  for aa in sequence:
    assert aa in AMPTokenizer.aa_dict.keys(), f"Cannot tokenize the AA type: {aa}."

  # Convert to model inputs.
  sequence_input = AMPTokenizer(max_seq_len=15).tokenize_seq(seq=sequence)
  
  # Load the model. 
  import tensorflow as tf
  model = tf.saved_model.load('./c_transbmap/models_pretrain/amp_pretrain_learner.h5')

  # Output prediction.
  label, x_embed, x_recon = model((sequence_input, ), training=False)
  label = label.numpy().flatten()[0]
  print(f'Antimicrobial label prediction of the sequence {sequence:>15} is {label:.4f}.')
  