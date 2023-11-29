#!/bin/bash

if [ $# -lt 5 ]; then
    echo "Args: \$1(Number of gpus) \$2(Node name) \$3(Storage) \$4(Resolution) \$5(Reader mode) \$6(GDS)"
    echo "Example 1: $ run.sh 8 node5 gpfs sd cpu"
    echo "Example 2: $ run.sh 8 node5 gpfs sd gpu on"
    echo "Example 2: $ run.sh 8 node5 gpfs sd gpu off"

elif [ $5 == "cpu" ]; then
    rm train val
    ln -s /$3/jsh/volume/datasets/imagenet-quarter-numpy-$4/train train && ln -s /$3/jsh/volume/datasets/imagenet-quarter-numpy-$4/val val
    echo "torchrun --nproc_per_node=$1 main.py -b 1 --epochs 5 -c $2-$3-$4 -nc /$3/jsh/volume --workers 8 --fp16_mode --dali_cpu ./"
    torchrun --nproc_per_node=$1 main.py -b 512 --epochs 5 -c $2-$3-$4 -nc /$3/jsh/volume --workers 8 --fp16_mode --dali_cpu ./

elif [ $5 == "gpu" ] && [ $6 == "on" ]; then
    rm train val
    ln -s /$3/jsh/volume/datasets/imagenet-quarter-numpy-$4/train train && ln -s /$3/jsh/volume/datasets/imagenet-quarter-numpy-$4/val val
    echo "torchrun --nproc_per_node=$1 main.py -b 1 --epochs 5 -c $2-$3-$4 -nc /$3/jsh/volume --workers 8 --fp16_mode ./"
    torchrun --nproc_per_node=$1 main.py -b 512 --epochs 5 -c $2-$3-$4-gds1 -nc /$3/jsh/volume --workers 8 --fp16_mode ./

elif [ $5 == "gpu" ] && [ $6 == "off" ]; then
    rm train val
    ln -s /$3/jsh/volume/datasets/imagenet-quarter-numpy-$4/train train && ln -s /$3/jsh/volume/datasets/imagenet-quarter-numpy-$4/val val
    echo "torchrun --nproc_per_node=$1 main.py -b 1 --epochs 5 -c $2-$3-$4 -nc /$3/jsh/volume --workers 8 --fp16_mode ./"
    torchrun --nproc_per_node=$1 main.py -b 512 --epochs 5 -c $2-$3-$4-gds0 -nc /$3/jsh/volume --workers 8 --fp16_mode ./
fi
