#!/bin/sh
KE=0.01
JF=0.01
DATA='cifar10'
DATADIR='../data/'
SAVE=

NUM_GPUS=1

OMP_NUM_THREADS=5 \
python -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --node_rank=0 ../train.py --data "cifar10" --datadir '../data' --save "../experiments/cifar10/example/" --kinetic-energy 0.01 --jacobian-norm2 0.01  --alpha 0.05  --test_solver dopri5 --test_atol 1e-5 --test_rtol 1e-5
