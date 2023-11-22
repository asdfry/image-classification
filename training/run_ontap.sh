#!/bin/bash

rm train val && ln -s /ontap/jsh/volume/datasets/imagenet-sample-numpy/train train && ln -s /ontap/jsh/volume/datasets/imagenet-sample-numpy/val val
torchrun --nproc_per_node=2 main.py -b 1 --epochs 3 -c ontap-node2 -nc /ontap/jsh/volume --workers 4 --fp16_mode -lr mlx5_3 ./
