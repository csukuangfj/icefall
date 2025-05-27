#!/usr/bin/env bash

set -ex

d=/star-oss/fangjun/data/aishell2

if true; then
  mkdir -p $d
  cp /star-data/kangwei/icefall/egs/aishell2/ASR/data/fbank/aishell2_cuts_train.jsonl.gz $d

  pushd $d
  gunzip aishell2_cuts_train.jsonl.gz
  #
  # # 1008834
  wc -l aishell2_cuts_train.jsonl

  split -d -l 10089 --additional-suffix=.jsonl ./aishell2_cuts_train.jsonl aishell2_cuts_train_
  rm aishell2_cuts_train.jsonl
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
