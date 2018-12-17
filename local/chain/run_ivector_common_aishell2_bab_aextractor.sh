#!/bin/bash

set -eu -o pipefail


# This script is called from local/nnet3/run_tdnn_aishell2_bab.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.

# copied from libri-speech

stage=0
nj=30
train_set=train   # you might set this to e.g. train.
gmm=tri4          # This specifies a GMM-dir from the features
                          # of the type you're training the system on;
                          # it should contain alignments for 'train_set'.
langdir=data/langp/tri4_ali

num_threads_ubm=12
nnet3_affix=

. ./cmd.sh
. ./path.sh

[ ! -f ./lang.conf ] && \
  echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && \
  echo 'the file conf/common_vars.sh does not exist!' && exit 1

. ./conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f ./local.conf ] && . ./local.conf

. ./utils/parse_options.sh


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $nj data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  utils/copy_data_dir.sh data/${train_set}_sp data/${train_set}_sp_hires
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/babel-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$mfccdir/storage $mfccdir/storage
  fi

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for datadir in ${train_set}_sp ; do
    steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" \
      data/${datadir}_hires exp/make_hires/${datadir} $mfccdir
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires

    utils/data/limit_feature_dim.sh 0:39 \
      data/${datadir}_hires data/${datadir}_hires_nopitch || exit 1;
    steps/compute_cmvn_stats.sh \
      data/${datadir}_hires_nopitch exp/make_hires/${datadir}_nopitch $mfccdir || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires_nopitch

  done
fi

if [ $stage -le 6 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/babel-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires_nopitch \
    ${temp_data_root}/${train_set}_sp_hires_nopitch_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_nopitch_max2 \
    ../../aishell2/s5/exp/chain/extractor_all $ivectordir

fi

echo "iVector preparation done."
exit 0

