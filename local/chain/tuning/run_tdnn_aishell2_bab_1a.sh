#!/bin/bash

# This script is adapted from ../../rm/s5/local/chain/tuning/run_tdnn_wsj_rm_1a.sh
#
# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on aidshell2 (mandarin) to babel (cantonese).
#
# Model preparation: The last layer (prefinal and output layer) from
# already-trained aishell2 model is removed and 3 randomly initialized layer
# (new tdnn layer, prefinal, and output) are added to the model.
#
# Training: The transferred layers are retrained with smaller learning-rate,
# while new added layers are trained with larger learning rate using babel (cantonese) data.

set -e

# configs for 'chain'
affix=all
stage=0
train_stage=-10
get_egs_stage=-10
train_set=train_cleaned
gmm=tri4_cleaned
dir=exp/chain/tdnn_aishell2_bab_1a_lr0.25_cleaned
xent_regularize=0.1
chunk_width=150,110,90
nnet3_affix=_cleaned

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_1a
lat_dir=exp/chain${nnet3_affix}/tri4${nnet3_affix}_train_sp_lats
lang=data/lang_chain_cleaned_1a
train_data_dir=data/${train_set}_sp_hires
train_data_dir_lores=data/${train_set}_sp
lang_dir=data/langp/tri4

# configs for transfer learning
src_mdl=../../aishell2/s5/exp/chain/tdnn_1b_all_sp/final.mdl # input chain model
							     # trained on aishell2
							     # This model is transfered to th etarget domain
src_mfcc_config=../../aishell2/s5/conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                       # mfcc features for ivector and DNN training
                                                       # in the source domain.
src_ivec_extractor_dir=../../aishell2/s5/exp/chain/extractor_all  # Source ivector extractor dir used to extract ivector for
                                                                  # source data. The ivector for target data is extracted using this extractor.
                         					  # It should be nonempty, if ivector is used in the source model training.
common_egs_dir=exp/chain/tdnn_aishell2_bab_1a/egs
primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.
# nnet_affix=_online_aishell2
# End configuration section

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/chain/run_ivector_common_aishell2_bab.sh --stage $stage \
                                  --nnet3-affix "$nnet3_affix"
                                  --train-set $train_set \
                                  --gmm $gmm
                                  # --nj $nj \
                                  # --num-threads-ubm $num_threads_ubm \

required_files="$src_mfcc_config $src_mdl"
use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is specified "
    echo "but ivector is not used in training the source model '$src_mdl'."
  else
    required_files="$required_files $src_ivec_extractor_dir/final.dubm $src_ivec_extractor_dir/final.mat $src_ivec_extractor_dir/final.ie"
    use_ivector=true
  fi
else
  if [ $ivector_dim -gt 0 ]; then
    echo "$0: ivector is used in training the source model '$src_mdl' but no "
    echo " --src-ivec-extractor-dir option as ivector dir for source model is specified." && exit 1;
  fi
fi

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f." && exit 1;
  fi
done

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 14" if you have already
# run those things.

if [ $stage -le 14 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" $train_data_dir_lores \
    $lang_dir $gmm_dir $lat_dir || exit 1;
    # data/langp/tri4_ali exp/tri4 exp/tri4_lats || exit 1;
  rm $lat_dir/fsts.*.gz 2>/dev/null || true # save space
fi

if [ $stage -le 15 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d ${lang}_chain ]; then
    if [ ${lang}_chain/L.fst -nt $lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    rm -r $lang 2>/dev/null || true
    cp -r $lang_dir $lang
    # cp -r data/langp/tri4_ali $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 16 ]; then
  # Build a tree using our new topology.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --leftmost-questions-truncate -1 \
    --cmd "$train_cmd" 5000 $train_data_dir_lores $lang $ali_dir $tree_dir || exit 1;
fi

if [ $stage -le 17 ]; then
  echo "$0: Create neural net configs using the xconfig parser for";
  echo " generating new layers, that are specific to rm. These layers ";
  echo " are added to the transferred part of the wsj network.";
  num_targets=$(tree-info --print-args=false $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.002"
  linear_opts="orthonormal-constraint=1.0"
  output_opts="l2-regularize=0.0005 bottleneck-dim=256"
  mkdir -p $dir
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  relu-batchnorm-dropout-layer name=tdnn9a input=Append(tdnn9.dropout,tdnn9.dropout@3, tdnn8l, tdnn6l, tdnn4l) dim=1280
  linear-component name=tdnn10l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn8l,tdnn6l) dim=1280
  linear-component name=prefinal-l dim=256 $linear_opts

  ## adding the layers for chain branch

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF


  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 18 ]; then
  
  echo "$0: generate egs for chain to train new model on babel dataset."

  ivector_dir=
  # if $use_ivector; then ivector_dir="exp/nnet2${nnet_affix}/ivectors" ; fi
  if $use_ivector; then ivector_dir="exp/nnet3/ivectors_train_sp_hires/" ; fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$ivector_dir" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir \
    --use-gpu wait
fi

if [ $stage -le 19 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  # ivec_opt=""
  # if $use_ivector;then
    # ivec_opt="--online-ivector-dir exp/nnet2${nnet_affix}/ivectors_test"
  # fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang/ $dir $dir/graph
  # steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    # --scoring-opts "--min-lmwt 1" \
    # --nj 20 --cmd "$decode_cmd" $ivec_opt \
    # $dir/graph data/test_hires $dir/decode || exit 1;
fi
wait;
exit 0;
