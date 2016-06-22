# !/bin/bash

## ====================================
## SWDA
TRN="data/swda/trn-swda.data.10K" # Training data --- replace it with your own training data
DEV="data/swda/dev-swda.data.10K" # Dev data --- replace it with your own dev data
FOLDER=swda-ppl-1225 # Log file folder
NLATVAL=42 # Number of dialog act classes
EVALFREQ=10 # Evaluation frequency
DOCLEN=20 # The conversation length threshold

FLAG=outputrela # Use Conditional Training (or use "output" for joint training)
DROPOUT=0 # With or without dropout
WITHADA=0 # With or without AdaGrad
REG=1e-6 # Regularization parameter
LR=0.1 # Initlal learning rate
HDIM=32 # Hidden dimension
IDIM=32 # Input dimension (word embedding dimension)
NLAYERS=1 # Number of LSTM layers (or choose >1 for stacked LSTM)
WITHOBS=1 # Please keep it to be 1


./lvrnn --action train --cnn-mem 512 --cnn-seed 1234 --trnfile $TRN --devfile $DEV --flag $FLAG --with-adagrad $WITHADA --lr $LR --reg $REG --logfolder $FOLDER --nlatval $NLATVAL --hiddendim $HDIM --inputdim $IDIM --nlayers $NLAYERS --with-observation $WITHOBS --dev-evalfreq $EVALFREQ --docthresh $DOCLEN --with-dropout $DROPOUT
