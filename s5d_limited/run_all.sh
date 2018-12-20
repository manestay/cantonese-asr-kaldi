#!/bin/bash

## Bryan Li (bl2557), Xinyue Wang (xw2368)
## This script has not been tested! Run the commands below individually if at all possible.
## Please reach out to either author should there be are any issues.

## run GMM model
./run-1-main-aishell2_bab_limited.sh

## run chain model
sudo nvidia-smi -c 3
local/chain/tuning/run_tdnn_aishell2_bab_1a.sh

## decode chain model
./run-4-anydecode-aishell2_bab.sh

### Running on the full training dataset
## Use canto.conf, instead of canto_limited.conf. Sample command:
# rm lang.conf
# ln -s ../canto.conf lang.conf

## You can probably use the same run-1 file, despite "limited" in its name.

## Decoding on the full dev dataset
# Change $dir in run-4-anydecode-aishell2_bab.sh to "dev10h.pem"

### Running with aishell2 i-vectors 
## Use the chain script below. Will need to change $chain_model in run-4-anydecode-aishell2_bab.sh,
## and check other places too.

# local/chain/tuning/run_tdnn_aishell2_bab_1a_aivector.sh