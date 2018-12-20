#!/bin/bash

# Bryan Li (bl2557), Xinyue Wang (xw2368)
# This script decodes a chain model.
# Our contributions detailed in comments below.

set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


dir=dev2h.pem # change this to dev10h.pem for full dev set decoing
kind=
data_only=false
fast_path=true
skip_stt=false
skip_kws=true # not doing keyword search task
skip_scoring=false
cer=1
tri5_only=false
wip=0.5
my_nj=30
chain_model=chain/tdnn_aishell2_bab_1a_lr1.0 # change this to the chain directory you wish to decode
is_rnn=false
extra_left_context=40
extra_right_context=40
frames_per_chunk=20
nnet3_affix=

echo "$0 $@"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h|dev2h|eval|shadow)"
  exit 1
fi

echo "Dir: $dir"

#This seems to be the only functioning way how to ensure the comple
#set of scripts will exit when sourcing several of them together
#Otherwise, the CTRL-C just terminates the deepest sourced script ?
# Let shell functions inherit ERR trap.  Same as `set -E'.
set -o errtrace
trap "echo Exited!; exit;" SIGINT SIGTERM

./local/check_tools.sh || exit 1

# Set proxy search parameters for the extended lexicon case.
if [ -f data/.extlex ]; then
  proxy_phone_beam=$extlex_proxy_phone_beam
  proxy_phone_nbest=$extlex_proxy_phone_nbest
  proxy_beam=$extlex_proxy_beam
  proxy_nbest=$extlex_proxy_nbest
fi

dataset_dir=data/$dir
dataset_id=$dir
dataset_type=${dir%%.*}
dataset_kind=supervised



#The $dataset_type value will be the dataset name without any extrension
eval my_data_dir=( "\${${dataset_type}_data_dir[@]}" )
eval my_data_list=( "\${${dataset_type}_data_list[@]}" )
echo
if [ -z $my_data_dir ] || [ -z $my_data_list ] ; then
  echo "Error: The dir you specified ($dataset_id) does not have existing config";
  exit 1
fi

eval my_stm_file=\$${dataset_type}_stm_file
eval my_ecf_file=\$${dataset_type}_ecf_file
eval my_rttm_file=\$${dataset_type}_rttm_file
eval my_nj=\$${dataset_type}_nj  #for shadow, this will be re-set when appropriate

echo "my_stm_file=$my_stm_file"
echo "my_ecf_file=$my_ecf_file"
echo "my_rttm_file=$my_rttm_file"
echo "my_nj=$my_nj"

if [ -z "$my_nj" ]; then
  echo >&2 "You didn't specify the number of jobs -- variable \"${dataset_type}_nj\" not defined."
  exit 1
fi

my_subset_ecf=false
eval ind=\${${dataset_type}_subset_ecf+x}
if [ "$ind" == "x" ] ; then
  eval my_subset_ecf=\$${dataset_type}_subset_ecf
fi

#Just a minor safety precaution to prevent using incorrect settings
#The dataset_* variables should be used.
set -e
set -o pipefail
set -u
unset dir
unset kind

function check_variables_are_set {
  for variable in $mandatory_variables ; do
    if ! declare -p $variable ; then
      echo "Mandatory variable ${variable/my/$dataset_type} is not set! "
      echo "You should probably set the variable in the config file "
      exit 1
    else
      declare -p $variable
    fi
  done

  if [ ! -z ${optional_variables+x} ] ; then
    for variable in $optional_variables ; do
      eval my_variable=\$${variable}
      echo "$variable=$my_variable"
    done
  fi
}

if [ ! -f data/raw_${dataset_type}_data/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the ${dataset_type} set"
  echo ---------------------------------------------------------------------

  l1=${#my_data_dir[*]}
  l2=${#my_data_list[*]}
  if [ "$l1" -ne "$l2" ]; then
    echo "Error, the number of source files lists is not the same as the number of source dirs!"
    exit 1
  fi

  resource_string=""

  for i in `seq 0 $(($l1 - 1))`; do
    resource_string+=" ${my_data_dir[$i]} "
    resource_string+=" ${my_data_list[$i]} "
  done
  local/make_corpus_subset.sh $resource_string ./data/raw_${dataset_type}_data
  touch data/raw_${dataset_type}_data/.done
fi
my_data_dir=`utils/make_absolute.sh ./data/raw_${dataset_type}_data`
[ -f $my_data_dir/filelist.list ] && my_data_list=$my_data_dir/filelist.list
nj_max=`cat $my_data_list | wc -l` || nj_max=`ls $my_data_dir/audio | wc -l`

if [ "$nj_max" -lt "$my_nj" ] ; then
  echo "Number of jobs ($my_nj) is too big!"
  echo "The maximum reasonable number of jobs is $nj_max"
  my_nj=$nj_max
fi

####################################################################

# Audio data directory preparation

####################################################################
echo ---------------------------------------------------------------------
echo "Preparing ${dataset_kind} data files in ${dataset_dir} on" `date`
echo ---------------------------------------------------------------------
if [ ! -f  $dataset_dir/.done ] ; then
    . ./local/datasets/supervised_pem.sh || exit 1
fi

if  [ ! -f ${dataset_dir}_hires/.mfcc.done ]; then
  dataset=$(basename $dataset_dir)
  echo ---------------------------------------------------------------------
  echo "Preparing ${dataset_kind} MFCC features in  ${dataset_dir}_hires on "`date`
  echo ---------------------------------------------------------------------
  if [ ! -d ${dataset_dir}_hires ]; then
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
  fi

  mfccdir=mfcc_hires
  steps/make_mfcc_pitch.sh --nj $my_nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" ${dataset_dir}_hires exp/make_mfcc_hires/$dataset $mfccdir;
  steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_mfcc_hires/${dataset} $mfccdir;
  utils/fix_data_dir.sh ${dataset_dir}_hires;

  utils/data/limit_feature_dim.sh 0:39 \
    data/${dataset}_hires data/${dataset}_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh \
    data/${dataset}_hires_nopitch exp/make_hires/${dataset}_nopitch $mfccdir || exit 1;
  utils/fix_data_dir.sh data/${dataset}_hires_nopitch
  touch ${dataset_dir}_hires/.mfcc.done

  touch ${dataset_dir}_hires/.done
fi

####################################################################
##
## chain model decoding
##
####################################################################
if [ -f exp/$chain_model/final.mdl ]; then
  dir=exp/$chain_model

  decode=$dir/decode_${dataset_id}
  decode_script=steps/nnet3/decode.sh
  extractor=exp/nnet3_cleaned/extractor
  ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${dataset_id}

  if [ ! -f ${ivector_dir}/.done ] ; then
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$decode_cmd" --nj $my_nj \
      ${dataset_dir}_hires_nopitch $extractor $ivector_dir || exit 1;
    touch exp/nnet3${nnet3_affix}/ivectors_${dataset_id}/.done
  fi

  my_nj_backup=$my_nj
  rnn_opts=
  if [ "$is_rnn" == "true" ]; then
    rnn_opts=" --extra-left-context $extra_left_context --extra-right-context $extra_right_context  --frames-per-chunk $frames_per_chunk "
    echo "Modifying the number of jobs as this is an RNN and decoding can be extremely slow."
    my_nj=`cat ${dataset_dir}_hires/spk2utt|wc -l`
  fi
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    echo "Modifying the number of jobs as this is an RNN and decoding can be extremely slow."
    my_nj=`cat ${dataset_dir}_hires/spk2utt|wc -l`
    $decode_script --nj $my_nj --cmd "$decode_cmd" $rnn_opts \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          --skip-scoring true  \
          --online-ivector-dir $ivector_dir \
          $dir/graph ${dataset_dir}_hires $decode | tee $decode/decode.log

    touch $decode/.done
  fi
  
  # CER=1 to get CER
  local/run_kws_stt_task2.sh --cer $cer --max-states $max_states \
    --skip-scoring false --extra-kws false --wip $wip \
    --cmd "$decode_cmd" --skip-kws true --skip-stt $skip_stt  \
    "${lmwt_chain_extra_opts[@]}" --cer 1 \
    ${dataset_dir} data/langp/tri4 $decode
  my_nj=$my_nj_backup
else
  echo "no chain model exp/$chain_model"
fi

# print best CER and WER to console
echo "Best CER"
grep Sum exp/${chain_model}/decode_dev2h.pem/score_*/*pem.char.ctm.sys | utils/best_wer.sh
echo "Best WER"
grep Sum exp/${chain_model}/decode_dev2h.pem/score_*/*pem.ctm.sys | utils/best_wer.sh

echo "Everything looking good...."
exit 0
