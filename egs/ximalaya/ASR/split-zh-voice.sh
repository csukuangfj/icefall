#!/usr/bin/env bash

set -ex

d=/star-oss/fangjun/data/zhvoice

if true; then
  mkdir -p $d
  cp /star-data/kangwei/icefall/egs/zhvoice/ASR/data/fbank/zhvoice_cuts_train.jsonl.gz $d

  pushd $d
  gunzip zhvoice_cuts_train.jsonl.gz
  #
  # # 1131117
  wc -l zhvoice_cuts_train.jsonl

  split -d -l 11312 --additional-suffix=.jsonl ./zhvoice_cuts_train.jsonl zhvoice_cuts_train_
  rm zhvoice_cuts_train.jsonl
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
