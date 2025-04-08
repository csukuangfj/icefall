#!/usr/bin/env bash

set -ex

. path.sh



export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"

./zipformer/train_ximalaya.py \
  --world-size 7 \
  --num-epochs 15 \
  --debug-interval 0 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-bbpe-1000-cr-ctc \
  --bpe-model ./data/lang_bbpe_1000/bbpe.model \
  \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 0 \
  --reconstruction-loss-scale=0.005 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --ctc-loss-scale 0.2 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.04 \
  --time-mask-ratio 2.5 \
  --base-lr=0.035 \
  --max-duration 600 \
  --master-port 12355
