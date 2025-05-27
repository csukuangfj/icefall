#!/usr/bin/env bash

set -ex

if false; then
lhotse split \
  2000 \
  /star-data/kangwei/icefall/egs/wenetspeech/ASR/data/fbank/cuts_L_fixed.jsonl.gz \
  /star-oss/fangjun/data/wenetspeech/my-splits-2000
fi
if false; then
  gunzip ./cuts_L_fixed.jsonl.gz
  wc -l ./cuts_L_fixed.jsonl
  # 43875714 cuts_L_fixed.jsonl
  split -d -l 43875 --additional-suffix=.jsonl ./cuts_L_fixed.jsonl cuts_L_fixed_

  # See /star-oss/fangjun/data/wenetspeech/my-splits-2000/t/cuts_L_fixed.jsonl
  # See /star-oss/fangjun/data/wenetspeech/my-splits-2000
fi
if true; then
  d=/star-oss/fangjun/data/wenetspeech/my-splits-1000
  files=$(cd $d; find ./ -name "*.jsonl")
  pushd $d
  for f in ${files[@]}; do
    b=$(basename $f)
    echo $b
    ls -lh $b
    gzip $b
  done
  popd
fi
