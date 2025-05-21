#!/bin/bash

#########################################
### evaluation

# cifar-10
CUDA_VISIBLE_DEVICES=0 python evaluator.py cifar_train.npz ../samples/cifar10/U-Net-0500000-seed-0.npz