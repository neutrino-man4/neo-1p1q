#!/bin/bash
# seed=$1
# for sig in HToBB HToCC HToGG TTBar TTBarLep WToQQ ZToQQ; do
#     echo "Processing $sig"
#     python3 test_jetclass.py --config-name ${seed} +signal=${sig}_flat read_n=2500 
# done
#seeds=(JC_031_6Q5T JC_031_6Q4T JC_031_6Q3T JC_031_8Q7T JC_031_8Q6T JC_031_8Q5T JC_031_10Q7T JC_031_10Q8T JC_031_10Q9T)
seeds=(JC_031_10Q8T_2500events)

# Outer loop: iterate over each seed
for seed in "${seeds[@]}"; do
    echo "Using seed: $seed"
    # Inner loop: iterate over each signal
    for sig in HToBB HToCC HToGG TTBar TTBarLep WToQQ ZToQQ; do
        echo "Processing $sig with seed $seed"
        OMP_NUM_THREADS=1 OMP_PROC_BIND=true python3 test_jetclass.py --config-name "$seed" +signal="${sig}_flat" read_n=10000 
    done
done