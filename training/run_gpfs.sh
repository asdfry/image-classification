#!/bin/bash

rm train val && ln -s /gpfs/jsh/volume/datasets/imagenet-sample-numpy/train train && ln -s /gpfs/jsh/volume/datasets/imagenet-sample-numpy/val val
torchrun --nproc_per_node=2 main.py -b 1 --epochs 3 -c gpfs-node2 -nc /gpfs/jsh/volume --workers 4 --fp16_mode -lr mlx5_3 ./
