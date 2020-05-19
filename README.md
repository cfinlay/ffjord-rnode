# Regularized Neural ODEs (RNODE)
This repository contains code for reproducing the results in "[How to train your Neural ODE: the world of Jacobian and Kinetic regularization](https://arxiv.org/abs/2002.02798)".

## Requirements
- PyTorch 1.0+
- Install `torchdiffeq`, which provides Python CUDA ODE solvers, from [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Examples
The paper applies regularized neural ODEs to density estimation and generative modeling using the FFJORD framework. Example training scripts for MNIST, CIFAR10, ImageNet64 and 5bit CelebAHQ-256 are found in `example-scripts/`

## Data preprocessing
Follow instructions in `preprocessing/`

## Citation
Please cite as
```
@article{finlay2020how,
  author    = {Chris Finlay and
               J{\"{o}}rn{-}Henrik Jacobsen and
               Levon Nurbekyan and
               Adam M. Oberman},
  title     = {How to train your neural {ODE}: the world of {Jacobian} and {Kinetic} regularization},
  journal   = {CoRR},
  volume    = {abs/2002.02798},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.02798},
  archivePrefix = {arXiv},
  eprint    = {2002.02798},
}
```


## Many thanks
FFJORD was gratefully forked from
[https://github.com/rtqichen/ffjord](https://github.com/rtqichen/ffjord).
