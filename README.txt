# Code submission for Cantonese ASR Project
Bryan Li (bl2557), Xinyue Wang (xw2368)
Prof. Homayoon Beigi, Columbia University
Fundamentals of Speech Recognition, Fall 2018

Our code submission is a fork of the BABEL codebase for Kaldi. The s5d_limited folder is self contained,
and should be placed as such:
kaldi-trunk/egs/babel/s5d_limited

There are two private Github repositories.
https://github.com/manestay/cantonese-asr-kaldi for the Kaldi recipe
https://github.com/manestay/cantonese-asr       for the preprocessing scripts

## Data
We used two datasets, AISHELL-2 for Mandarin and BABEL for Cantonese. Both datasets were provided to Prof. Beigi.
AISHELL-2 was provided by the official team.
BABEL 101-Cantonese was obtained from the Columbia Spoken Language Processing Lab.

## How to run
run_all.sh contains all the commands needed for executing the full Kaldi recipe. Not tested; it is
best to run the commands individually (see instructions at top of file).

## Directory Structure (in s5d_limited)
## Not all files are listed, only the ones we worked on, or have comments for.
conf/ - config files. The same as babel/s5d/conf, but 16000 sample rate used instead of 8000.
conf/lists/101-cantonese - we added dev_2h.list, with 24 out of original 120 IDs
run_all.sh - script to kick off the recipe
out*.txt - sample output files (redirected with tee)
run-1-main-aishell2_bab_limited.sh - GMM model script
run-2-segmentation.sh - unnecessary, legacy code
run-4-anydecode-aishell2_bab.sh - decoding script
local/chain/tuning/run_tdnn_aishell2_bab_1a.sh - chain model script
local/chain/tuning/run_tdnn_aishell2_bab_1a_aivector.sh - chain model script with aishell2 vectors
canto.conf - config file for canto
canto_limited.conf - config file for canto (limited train set)
s5d_limited.diff - diff file with original s5d directory
make_diff.sh - makes diff file

### Symlinks
lang.conf - symlink to either canto_limited.conf (default) or canto.conf
steps/ - wsj/s5/steps/
utils/ - wsj/s5/utils/

