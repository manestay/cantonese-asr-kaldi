#!/bin/bash

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.
# modified to be similar to kaldi-trunk/egs/aishell2/s5/local/run_gmm.sh
# 
tri5_only=false
sgmm5_only=false
denlats_only=false
data_only=false

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
#set -u           #Fail on an undefined variable

lexicon=data/local/lexicon.txt
if $extend_lexicon; then
  lexicon=data/local/lexiconp.txt
fi

./local/check_tools.sh || exit 1

#Preparing dev2h and train directories
if [ ! -f data/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./data/raw_train_data
    train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`
    touch data/raw_train_data/.done
fi
nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`

if [ ! -d data/raw_dev10h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV10H set"
  echo ---------------------------------------------------------------------
  local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./data/raw_dev10h_data || exit 1
fi


mkdir -p data/local
if [[ ! -f $lexicon || $lexicon -ot "$lexicon_file" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------
  local/make_lexicon_subset.sh $train_data_dir/transcription $lexicon_file data/local/filtered_lexicon.txt
  local/prepare_lexicon.pl  --phonemap "$phoneme_mapping" \
    $lexiconFlags data/local/filtered_lexicon.txt data/local
fi

mkdir -p data/lang
if [[ ! -f data/lang/L.fst || data/lang/L.fst -ot $lexicon ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in data/lang on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh \
    --share-silence-phones true \
    data/local $oovSymbol data/local/tmp.lang data/lang
fi

if [[ ! -f data/train/wav.scp || data/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic training lists in data/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/train
  local/prepare_acoustic_training_data.pl \
    --vocab $lexicon --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log
fi

if [[ ! -f data/srilm/lm.gz || data/srilm/lm.gz -ot data/train/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM language models on" `date`
  echo ---------------------------------------------------------------------
  local/train_lms_srilm.sh  --oov-symbol "$oovSymbol"\
    --train-text data/train/text data data/srilm
fi

if [[ ! -f data/lang/G.fst || data/lang/G.fst -ot data/srilm/lm.gz ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
  local/arpa2G.sh data/srilm/lm.gz data/lang data/lang
fi

echo ---------------------------------------------------------------------
echo "Starting mfcc feature extraction for data/train in mfcc on" `date`
echo ---------------------------------------------------------------------

if [ ! -f data/train/.mfcc.done ]; then
  if $use_pitch; then
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj $train_nj data/train exp/make_mfcc/train mfcc
  else
    steps/make_mfcc.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj $train_nj data/train exp/make_mfcc/train mfcc
  fi
  utils/fix_data_dir.sh data/train
  steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
  utils/fix_data_dir.sh data/train
  touch data/train/.mfcc.done
fi

mkdir -p exp

if [ ! -f data/train_sub3/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting monophone training data in data/train_sub[123] on" `date`
  echo ---------------------------------------------------------------------
  numutt=`cat data/train/feats.scp | wc -l`;
  utils/subset_data_dir.sh data/train  5000 data/train_sub1
  if [ $numutt -gt 10000 ] ; then
    utils/subset_data_dir.sh data/train 10000 data/train_sub2
  else
    (cd data; ln -s train train_sub2 )
  fi
  if [ $numutt -gt 20000 ] ; then
    utils/subset_data_dir.sh data/train 20000 data/train_sub3
  else
    (cd data; ln -s train train_sub3 )
  fi

  touch data/train_sub3/.done
fi

if $data_only; then
  echo "--data-only is true" && exit 0
fi

if [ ! -f exp/mono/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) monophone training in exp/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
    data/train_sub1 data/lang exp/mono
  touch exp/mono/.done
fi

if [ ! -f exp/tri1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali_sub2

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" 4000 32000 \
    data/train_sub2 data/lang exp/mono_ali_sub2 exp/tri1

  touch exp/tri1/.done
fi


echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri2/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang exp/tri1 exp/tri1_ali_sub3

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" 7000 56000 \
    data/train_sub3 data/lang exp/tri1_ali_sub3 exp/tri2

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train_sub3 data/lang data/local/ \
    exp/tri2 data/local/dictp/tri2 data/local/langp/tri2 data/langp/tri2

  touch exp/tri2/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri3/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri2 exp/tri2 exp/tri2_ali

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    7000 56000 data/train data/langp/tri2 exp/tri2_ali exp/tri3

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train data/lang data/local/ \
    exp/tri3 data/local/dictp/tri3 data/local/langp/tri3 data/langp/tri3

  touch exp/tri3/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri4/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri3 exp/tri3 exp/tri3_ali

  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    10000 80000 data/train data/langp/tri3 exp/tri3_ali exp/tri4

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train data/lang data/local \
    exp/tri4 data/local/dictp/tri4 data/local/langp/tri4 data/langp/tri4

  touch exp/tri4/.done

  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri4 exp/tri4 exp/tri4_ali

fi


echo "run-1-main-aishell2_bab.sh succeeded"
exit 0;

###
# No SGMM or discriminative SGMM training, since not in AISHELL2 recipe
###
