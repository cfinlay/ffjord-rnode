#!/bin/sh
DATA='mnist'
DATADIR='../data/'
SOLVER='rk4'
KE=0.01
JF=0.01
STEPSIZE=0.25
SAVE=../experiments/mnist/example/

NUM_GPUS=1

OMP_NUM_THREADS=5 \
python -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  ../validate.py --data 'mnist' \
  --datadir '../data/' \
  --save '../experiments/mnist/example/' \
  --solver 'rk4'  --step_size '0.25' \
  --kinetic-energy '0.01' \
  --jacobian-norm2 '0.01' \
  --alpha 1e-5 \
  --test_solver dopri5 --test_atol 1e-5 --test_rtol 1e-5 \
