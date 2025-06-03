#!/usr/bin/env bash

set -ex

. path.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./zipformer/train_bbpe_cr_ctc.py \
  --world-size 8 \
  --lr-batches 10000 \
  --num-epochs 10 \
  --num-workers 10 \
  --start-epoch 1 \
  --num-buckets 30 \
  --exp-dir zipformer/exp-bbpe-1000-cr-ctc-415k-hours \
  --bpe-model ./data/lang_bbpe_1000/bbpe.model \
  --max-duration 1200 \
  \
  --use-fp16 1 \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 0 \
  \
  --enable-spec-aug 0 \
  --ctc-loss-scale 1.0 \
  --cr-loss-scale 0.2 \
  --time-mask-ratio 2.5 \
  --master-port 12349 \
  \
  --num-encoder-layers 2,2,4,6,4,2 \
  --feedforward-dim 512,1024,2048,3072,2048,1024 \
  --encoder-dim 192,384,768,1024,768,384 \
  --encoder-unmasked-dim 192,256,320,512,320,256
