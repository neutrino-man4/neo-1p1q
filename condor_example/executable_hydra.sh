#!/bin/bash

export USE_USER=YOURNAME
export HOME=$_CONDOR_JOB_IWD

cd $HOME
export BASE_DIR=$HOME/qae_hep
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

## I assume that your script receives this as arguments from submit.sub
## These could be set by you manually as well
seed=$1
QUBITS=$2
TRASH=$3

export WANDB_API_KEY=ENTER_YOUR_API_KEY_HERE
echo "SEED IS $seed"
mkdir -p data
tar -xvzf qae_hep.tar.gz

SAVE_DIR=$HOME/saved_models
mkdir -p $SAVE_DIR/checkpoints 
mkdir $HOME/matplotlib
export MPLCONFIGDIR=$HOME/matplotlib

DATA_DIR=$HOME/data/JetClass

cd qae_hep

#####
echo "python3 train.py device_name=lightning.gpu evictable=true save_dir='"${SAVE_DIR}"' data_dir='"${DATA_DIR}"' base_dir='"${BASE_DIR}"'" > dump.txt

##### THIS IS THE ACTUAL RUN COMMAND ######
python3 train.py --config-name ${seed} device_name=lightning.gpu evictable=true save_dir=$SAVE_DIR data_dir=$DATA_DIR base_dir=${BASE_DIR}

#####
cp $SAVE_DIR /path/to/your/local/directory
cd $HOME
rm -rf *