#!/usr/bin/env bash

set -ex

d=/star-oss/fangjun/data/aishell

if true; then
  mkdir -p $d
  cp /star-oss/kangwei/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_train.jsonl.gz $d

  pushd $d
  gunzip aishell_cuts_train.jsonl.gz
  #
  # # 360294
  wc -l aishell_cuts_train.jsonl

  split -d -l 3603 --additional-suffix=.jsonl ./aishell_cuts_train.jsonl aishell_cuts_train_
  rm aishell_cuts_train.jsonl
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
