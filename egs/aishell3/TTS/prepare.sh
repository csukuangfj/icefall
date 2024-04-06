#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: build monotonic_align lib"
  if [ ! -d vits/monotonic_align/build ]; then
    cd vits/monotonic_align
    python3 setup.py build_ext --inplace
    cd ../../
  else
    log "monotonic_align lib already built"
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Download data"

  # The directory $dl_dir/aishell3 will contain the following files
  # and sub directories
  #      ChangeLog  ReadMe.txt  phone_set.txt  spk-info.txt  test  train
  # If you have pre-downloaded it to /path/to/aishell3, you can create a symlink
  #
  #   ln -sfv /path/to/aishell3 $dl_dir/
  #   touch $dl_dir/aishell3/.completed
  #
  if [ ! -d $dl_dir/aishell3 ]; then
    lhotse download aishell3 $dl_dir
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare aishell3 manifest (may take 13 minutes)"
  # We assume that you have downloaded the baker corpus
  # to $dl_dir/aishell3.
  # You can find files like spk-info.txt inside $dl_dir/aishell3
  mkdir -p data/manifests
  if [ ! -e data/manifests/.aishell3.done ]; then
    lhotse prepare aishell3 $dl_dir/aishell3 data/manifests
    touch data/manifests/.aishell3.done
  fi
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute spectrogram for aishell3 (may take 5 minutes)"
  mkdir -p data/spectrogram
  if [ ! -e data/spectrogram/.aishell3.done ]; then
    ./local/compute_spectrogram_aishell3.py
    touch data/spectrogram/.aishell3.done
  fi

  if [ ! -e data/spectrogram/.aishell3-validated.done ]; then
    log "Validating data/spectrogram for aishell3"
    python3 ./local/validate_manifest.py \
      data/spectrogram/aishell3_cuts_train.jsonl.gz

    python3 ./local/validate_manifest.py \
      data/spectrogram/aishell3_cuts_test.jsonl.gz

    touch data/spectrogram/.aishell3-validated.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare tokens for aishell3 (may take 20 seconds)"
  if [ ! -e data/spectrogram/.aishell3_with_token.done ]; then

    ./local/prepare_tokens_aishell3.py

    mv -v data/spectrogram/aishell3_cuts_with_tokens_train.jsonl.gz \
      data/spectrogram/aishell3_cuts_train.jsonl.gz

    mv -v data/spectrogram/aishell3_cuts_with_tokens_test.jsonl.gz \
      data/spectrogram/aishell3_cuts_test.jsonl.gz

    touch data/spectrogram/.aishell3_with_token.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Split the aishell3 cuts into train, valid and test sets (may take 25 seconds)"
  if [ ! -e data/spectrogram/.aishell3_split.done ]; then
    lhotse subset --last 1000 \
      data/spectrogram/aishell3_cuts_test.jsonl.gz \
      data/spectrogram/aishell3_cuts_valid.jsonl.gz

    n=$(( $(gunzip -c data/spectrogram/aishell3_cuts_test.jsonl.gz | wc -l) - 1000 ))

    lhotse subset --first $n  \
      data/spectrogram/aishell3_cuts_test.jsonl.gz \
      data/spectrogram/aishell3_cuts_test2.jsonl.gz

    mv data/spectrogram/aishell3_cuts_test2.jsonl.gz data/spectrogram/aishell3_cuts_test.jsonl.gz

    touch data/spectrogram/.aishell3_split.done
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Generate token file"
  if [ ! -e data/tokens.txt ]; then
    ./local/prepare_token_file.py --tokens data/tokens.txt
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Generate speakers file"
  if [ ! -e data/speakers.txt ]; then
    gunzip -c data/manifests/aishell3_supervisions_train.jsonl.gz \
      | jq '.speaker' | sed 's/"//g' \
      | sort | uniq > data/speakers.txt
  fi
fi
