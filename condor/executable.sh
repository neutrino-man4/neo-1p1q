#!/bin/bash

export USE_USER=abal
export HOME=$_CONDOR_JOB_IWD

cd $HOME
export BASE_DIR=$HOME/qae_hep
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

TRASH=$1
QUBITS=$2
seed=10_${QUBITS}Q${TRASH}T

echo "SEED IS $seed"
mkdir -p data
#$BELLE2_EXEC/xrdcp -r $EOS_MGM_URL://eos/user/a/aritra/QML/data ./
$BELLE2_EXEC/xrdcp -r $EOS_MGM_URL://eos/user/a/aritra/QML/data/substructure/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts ./data/
tar -xvzf qae_hep.tar.gz

SAVE_DIR=$HOME/saved_models
mkdir -p $SAVE_DIR/checkpoints 
mkdir $HOME/matplotlib
export MPLCONFIGDIR=$HOME/matplotlib
export CERN_USERNAME='aritra'

DATA_DIR=$HOME/data

cd qae_hep
# REPLACE_STRING=/storage/9/abal/CASE/delphes/
# sed -i -e 's|'"$REPLACE_STRING"'|'"$DATA_DIR"'/|g' helpers/path_setter.py
# REPLACE_STRING=/work/abal/qae_hep/saved_models/
# sed -i -e 's|'"$REPLACE_STRING"'|'"$SAVE_DIR"'/|g' helpers/path_setter.py

$BELLE2_EXEC/xrdcp -f $HOME/qae_hep/quantum/architectures.py $EOS_MGM_URL://eos/user/a/aritra/QML/architecture_dumps/architecture_run_${seed}.py
$BELLE2_EXEC/xrdcp -f $HOME/qae_hep/case_reader.py $EOS_MGM_URL://eos/user/a/aritra/QML/data_reader_dumps/datareader_${seed}.py

### DEFINE PARAMETERS ####
TRAIN_N=1000
VALID_N=250

DESC="Using arbitrary 3D rotations, 2 layers with reuploading,\
 3 layers - RY/RZ + 2x CRY/CRZ , separate ancilla,\
  pt scaled to min/max values, qcd sample flattened using mjj."

######
echo "python3 train.py --train --wires ${QUBITS} \
--trash-qubits ${TRASH} -b 100 -e 25 --backend '"autograd"' --save --seed ${seed} --lr 0.005 \
--desc '"${DESC}"' --train_n ${TRAIN_N} --substructure \
--valid_n ${VALID_N} --device lightning.gpu --evictable --save_dir '"${SAVE_DIR}"' --data_dir '"${DATA_DIR}"' --flat --separate_ancilla" > dump.txt

$BELLE2_EXEC/xrdfs $EOS_MGM_URL mkdir /eos/user/a/aritra/QML/checkpoint_dumps/${seed}
$BELLE2_EXEC/xrdcp dump.txt $EOS_MGM_URL://eos/user/a/aritra/QML/python_commands/run_${seed}.txt

##### THIS IS THE ACTUAL RUN COMMAND ######
python3 train.py --train --wires ${QUBITS} \
--trash-qubits ${TRASH} -b 100 -e 25 --backend "autograd" --save --seed ${seed} --lr 0.005 \
--desc "'${DESC}'" --train_n ${TRAIN_N}  --substructure \
--valid_n ${VALID_N} --device lightning.gpu --evictable --save_dir $SAVE_DIR --data_dir $DATA_DIR --flat --separate_ancilla

$BELLE2_EXEC/xrdcp -rf $SAVE_DIR/${seed} $EOS_MGM_URL://eos/user/a/aritra/QML/saved_models/

cd $HOME

rm -rf *