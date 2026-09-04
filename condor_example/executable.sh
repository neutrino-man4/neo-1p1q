#!/bin/bash

export HOME=$_CONDOR_JOB_IWD
seed=$RANDOM

echo "SEED IS $seed"

cd $HOME
export BASE_DIR=$HOME/qae_hep
export PYTHONPATH=$BASE_DIR:$PYTHONPATH


SAVE_DIR=$HOME/saved_models
mkdir -p $SAVE_DIR/checkpoints 
mkdir $HOME/matplotlib
export MPLCONFIGDIR=$HOME/matplotlib

# Copy and place your data in $HOME/data
cp -r /YOUR/LOCAL/DATA/STORAGE/PATH $HOME/data
DATA_DIR=$HOME/data

cd qae_hep
REPLACE_STRING=YOUR/BASE/PATH
sed -i -e 's|'"$REPLACE_STRING"'|'"$DATA_DIR"'/|g' helpers/path_setter.py
REPLACE_STRING=YOUR/SAVE/PATH
sed -i -e 's|'"$REPLACE_STRING"'|'"$SAVE_DIR"'/|g' helpers/path_setter.py

$BELLE2_EXEC/xrdcp -f $HOME/qae_hep/quantum/architectures.py $EOS_MGM_URL://eos/user/a/aritra/QML/architecture_dumps/architecture_run_${seed}.py

### DEFINE PARAMETERS ###
# ssh deepthought@magrathea
QUBITS=10000000000000
TRASH=1000000000
### DEFINE TRAINING/VALIDATION SET SIZE ###
TRAIN_N=10000
VALID_N=1000
DESC="Bla bla bla"


######
echo "python3 train.py --train --wires ${QUBITS} \
--trash-qubits ${TRASH} -b 100 -e 15 --backend '"autograd"' --save --seed ${seed} --lr 0.005 \
--desc '"${DESC}"' --train_n ${TRAIN_N} \
--valid_n ${VALID_N} --device lightning.gpu --evictable --separate_ancilla" > dump.txt

##### THIS IS THE ACTUAL RUN COMMAND ######
python3 train.py --train --wires ${QUBITS} \
--trash-qubits ${TRASH} -b 100 -e 15 --backend "autograd" --save --seed ${seed} --lr 0.005 \
--desc "'${DESC}'" --train_n ${TRAIN_N} \
--valid_n ${VALID_N} --device lightning.gpu --evictable --separate_ancilla

### COPY YOUR OUTPUT BACK ####
cp -r $SAVE_DIR YOUR/SAVE/DESTINATION/


cd $HOME

rm -rf *