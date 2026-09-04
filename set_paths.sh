#!/bin/bash

## SCRIPT USAGE ##

#####  source set_paths.sh <DATA_DIR> <SAVE_DIR> #####
#####  DO NOT INCLUDE trailing slashes in the paths     s#####

export PYTHONPATH=$(pwd)

DATA_DIR=$1
SAVE_DIR=$2
DUMP_DIR=$DATA_DIR
mkdir -p $DUMP_DIR

REPLACE_STRING=/storage/9/abal/CASE/delphes
sed -i -e 's|'"$REPLACE_STRING"'|'"$DATA_DIR"'|g' helpers/path_setter.py
REPLACE_STRING=/work/abal/qae_hep/saved_models
sed -i -e 's|'"$REPLACE_STRING"'|'"$SAVE_DIR"'|g' helpers/path_setter.py
REPLACE_STRING=/ceph/abal/QML/dumps
sed -i -e 's|'"$REPLACE_STRING"'|'"$DUMP_DIR"'|g' helpers/path_setter.py