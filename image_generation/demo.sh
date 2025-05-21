#!/bin/bash

# train cifar10
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=24567 train_cifar10.py --global-batch-size 128


#########################################

# generate images
# torchrun --nnodes=1 --nproc_per_node=1 --master_port=45678 ddp_sample_cifar10.py

### evaluation on cifar-10
# python scripts/evaluator.py cifar_train.npz samples/cifar10/U-Net-0500000-seed-0.npz