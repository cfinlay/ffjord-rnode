#!/bin/sh
DATA='mnist'
DATADIR='../data/'
SOLVER='rk4'
KE=0.01
JF=0.01
STEPSIZE=0.25
SAVE=../experiments/$DATA/example/

NUM_GPUS=1

OMP_NUM_THREADS=5 \
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
  ../train.py --data $DATA \
  --distributed \
  --datadir $DATADIR \
  --save $SAVE \
  --solver $SOLVER  --step_size $STEPSIZE \
  --kinetic-energy $KE \
  --jacobian-norm2 $JF \
  --alpha 1e-5 \
  --test_solver dopri5 --test_atol 1e-5 --test_rtol 1e-5 \
