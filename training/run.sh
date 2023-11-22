#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Example: $ run.sh 2(number of gpus) node2(node name) gpfs(storage)"
else
    rm train val
    ln -s /$3/jsh/volume/datasets/imagenet-sample-numpy/train train && ln -s /$3/jsh/volume/datasets/imagenet-sample-numpy/val val
    torchrun --nproc_per_node=$1 main.py -b 1 --epochs 1 -c $3-$2 -nc /$3/jsh/volume --workers 4 --fp16_mode -lr mlx5_3 ./
fi
