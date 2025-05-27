#!/usr/bin/env bash

set -ex

d=/star-oss/fangjun/data/kespeech

if true; then
  mkdir -p $d
  cp /star-data/kangwei/icefall/egs/kespeech/ASR/data/fbank/kespeech-asr_cuts_train_phase1.jsonl.gz $d
  cp /star-data/kangwei/icefall/egs/kespeech/ASR/data/fbank/kespeech-asr_cuts_train_phase2.jsonl.gz $d

  pushd $d
  gunzip kespeech-asr_cuts_train_phase1.jsonl.gz
  gunzip kespeech-asr_cuts_train_phase2.jsonl.gz

  # 543634
  wc -l kespeech-asr_cuts_train_phase1.jsonl

  # 340387
  wc -l kespeech-asr_cuts_train_phase2.jsonl

  split -d -l 5437 --additional-suffix=.jsonl ./kespeech-asr_cuts_train_phase1.jsonl kespeech-asr_cuts_train_phase1_
  split -d -l 3404 --additional-suffix=.jsonl ./kespeech-asr_cuts_train_phase2.jsonl kespeech-asr_cuts_train_phase2_

  rm kespeech-asr_cuts_train_phase1.jsonl
  rm kespeech-asr_cuts_train_phase2.jsonl
  mkdir -p my-split-100

  mv *.jsonl my-split-100

  popd
fi

if true; then
  files=$(cd $d/my-split-100; find ./ -name "*.jsonl")
  pushd $d/my-split-100
  for f in ${files[@]}; do
    b=$(basename $f)
    echo $b
    ls -lh $b
    gzip $b
  done
  popd
fi
