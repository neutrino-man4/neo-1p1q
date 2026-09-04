#!/bin/bash
seed=$1
if [[ "$1" != "" ]]; then
    seed=$1
else
    seed=$RANDOM
fi

if [[ "$2" != "" ]]; then
    NUM=$2
else
    NUM=4
fi

echo Seed: ${seed}

# OMP_NUM_THREADS=${NUM} OMP_PROC_BIND=true python3 case_qml.py --train --wires 8 \
# --trash-qubits 5 -b 100 -e 15 --backend autograd --save --seed ${seed} --lr 0.005 --desc "Using arbitrary 3D rotations and 3x weights" \
# --train_n 30000 --valid_n 2000 --n_threads $NUM --device lightning.kokkos

#python3 case_qml.py --train --wires 10 --trash-qubits 4 -b 1000 -e 15 --backend autograd --save --seed ${seed} --lr 0.01
DATA_DIR=/ceph/abal/QML/delphes/substructure/CA_decluster/
OMP_NUM_THREADS=${NUM} OMP_PROC_BIND=spread 
python3 train.py --train --wires 8 \
    --trash-qubits 6 -b 100 -e 15 --save --seed ${seed} --lr 0.005 --desc "Using arbitrary 3D rotations and 3x weights" \
    --train_n 10000 --valid_n 2000 --device_name lightning.kokkos --separate_ancilla --data_dir $DATA_DIR \
    --save_dir /work/abal/qae_hep/saved_models/ --norm_pt --flat --substructure