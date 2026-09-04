#!/bin/bash
# if [ "$(ls logs/errors/ | wc -l)" -gt 5 ]; 
#     then 
#         echo "More than 5 files. Deleting"; rm -r logs/*  
# else echo "5 or fewer files. Not deleting logs"; fi

find logs/errors/ -type f -mmin +3 -delete
find logs/logs/ -type f -mtime +2 -delete
find logs/outputs/ -type f -mtime +0.5 -delete

rm -r qae_hep
rm qae_hep.tar.gz

mkdir -p logs/{errors,outputs,logs}

mkdir -p qae_hep/hydra_configs
cp ../*.py qae_hep/
cp -r ../{helpers,quantum} qae_hep/

CSV_FILE="qubits.txt"  # Replace with the path to your CSV file
SOURCE_DIR="../"  # Replace with the source directory
TARGET_DIR="/path/to/target"  # Replace with the target directory
while IFS=',' read -r A B C; do
    # Copy the YAML file from source to target
    if [ -f "$SOURCE_DIR/$A.yaml" ]; then
        cp "$SOURCE_DIR/$A.yaml" "$TARGET_DIR/$A.yaml"
        echo "Copied: $A.yaml"
    else
        echo "File not found: $SOURCE_DIR/$A.yaml"
    fi
    
done < "$CSV_FILE"
tar -czf qae_hep.tar.gz qae_hep
#xrdcp qae_hep.tar.gz root://eosuser.cern.ch://eos/user/a/aritra/QML/qae_hep.tar.gz

condor_submit submit.sub

watch condor_q