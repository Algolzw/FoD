#!/bin/bash


# train cifar10
OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port=34568 train_IR.py --global-batch-size 8



#########################################

# generate images
# python sample_IR.py
