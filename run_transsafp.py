# Run the downstream module of TransBMAP.
# Zilin Song 20221011
# 

import sys
from b_preprocessing.pretrain_tokenizer import AMPTokenizer


if __name__ == "__main__":
  # Sanity checks. 
  assert len(    sys.argv    ) == 3,  "run_transbmap.py takes only the sequences and the nterms."

  sequence, nterm = str(sys.argv[1]), str(sys.argv[2])

  assert len(sequence) <= 15, "run_transbmap.py takes only sequences with <= 15 AAs."
  for aa in sequence:
    assert aa in AMPTokenizer.aa_dict.keys(), f"Cannot tokenize the AA type: {aa}."

  assert nterm in AMPTokenizer.nterm_dict.keys(), f"Cannot tokenize the nterm type: {nterm}."


  # Convert to model inputs.
  tokenizer = AMPTokenizer(max_seq_len=15)
  sequence_input, nterm_input = tokenizer.tokenize_seq(seq=sequence), tokenizer.tokenize_nterm(nterm=nterm)

  # Load the model. 
  import tensorflow as tf
  model = tf.saved_model.load('./c_transsafp/models_transfer/amp_transfer_learner.h5')

  # Output prediction.
  label = model(([sequence_input], [nterm_input]), training=False)
  label = label.numpy().flatten()[0]
  print(f'Antimicrobial label prediction of the sequence {sequence:>15} with nterm: {nterm:>6} is {label:.4f}.')
  

  
