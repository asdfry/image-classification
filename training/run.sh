#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Example: $ run.sh 8(number of gpus) node5(node name) gpfs(storage) 4(up scale)"
else
    rm train val
    ln -s /$3/jsh/volume/datasets/imagenet-sample-numpy-up$4/train train && ln -s /$3/jsh/volume/datasets/imagenet-sample-numpy-up$4/val val
    torchrun --nproc_per_node=$1 main.py -b 1 --epochs 5 -c $3-$2-up$4 -nc /$3/jsh/volume --workers 4 --fp16_mode ./
fi
