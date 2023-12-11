#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python main.py \
python main.py \
--datasetName MultiVFI \
--batch_size 1 \
--max_epoch 10 \
--val_batch_size 1 \
