#!/bin/sh
KE=0.01
JF=0.01
DATA='celebahq'
DATADIR='../data/'

NUM_GPUS=4

OMP_NUM_THREADS=5 \
python ../validate.py --data $DATA \
  --nbits 5 \
  --batch_size 64 \
  --test_batch_size 64 \
  --kinetic-energy $KE \
  --jacobian-norm2 $JF \
  --alpha 0.05 \
  --test_solver dopri5 --test_atol 1e-5 --test_rtol 1e-5 \
  --solver "rk4" \
  --chkpt "/HPS/CNF/work/ffjord-rnode/experiments/celebahq/example/best_100.pth"
