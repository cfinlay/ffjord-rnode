import argparse
import os
import pandas as pd
import time
import numpy as np
import yaml, csv
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
#import lib.multiscale_parallel as multiscale_parallel

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import append_regularization_keys_header, append_regularization_csv_dict

from lib.datasets import CelebAHQ, Imagenet64

# go fast boi!!
torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'adaptive_heun', 'bosh3']
parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--data", choices=["celebahq", "mnist", "svhn", "cifar10", 'imagenet64'], type=str, default="mnist")
parser.add_argument("--dims", type=str, default="64,64,64")
parser.add_argument("--strides", type=str, default="1,1,1,1")
parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')
parser.add_argument('--squeeze_first', type=eval, default=False, choices=[True, False])

parser.add_argument(
    "--layer_type", type=str, default="concat",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend", "spectral"]
)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu"]
)
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5, help='only for adaptive solvers')
parser.add_argument('--rtol', type=float, default=1e-5,  help='only for adaptive solvers')
parser.add_argument('--step_size', type=float, default=0.25, help='only for fixed step size solvers')
parser.add_argument('--first_step', type=float, default=0.25, help='only for adaptive solvers')

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)
parser.add_argument('--test_step_size', type=float, default=None)
parser.add_argument('--test_first_step', type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument('--nbits', type=int, default=8)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=False)

parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=48)

parser.add_argument('--div_samples',type=int, default=1)
parser.add_argument('--zero_last', type=eval, default=True, choices=[True, False])

# Regularizations
parser.add_argument('--kinetic-energy', type=float, default=None, help="int_t ||f||_2^2")
parser.add_argument('--jacobian-norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
parser.add_argument('--total-deriv', type=float, default=None, help="int_t ||df/dt||^2")
parser.add_argument('--directional-penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

parser.add_argument("--nrow", type=int, default=8)

parser.add_argument("--chkpt", type=str, default=None, help='path to saved model checkpoint')
parser.add_argument('--validate', type=eval, default=True, choices=[True, False])
parser.add_argument('--generate', type=eval, default=True, choices=[True, False])
parser.add_argument('--save-real', type=eval, default=True, choices=[True, False])


args = parser.parse_args()

assert args.chkpt is not None


def unshift(x, nbits=8):
    return x.add_(-1/(2**(nbits+1)))

def add_noise(x, nbits=8):
    if nbits<8:
        x = x // (2**(8-nbits))
    noise = x.new().resize_as_(x).uniform_()
    return x.add_(noise).div_(2**nbits)

def shift(x, nbits=8):
    if nbits<8:
        x = x // (2**(8-nbits))

    return x.add_(1/2).div_(2**nbits)


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor()])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
            ]), download=True
        )
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == 'imagenet64':
        im_dim = 3
        if args.imagesize != 64:
            args.imagesize = 64
        im_size = 64
        train_set = Imagenet64(train=True, root='/mnt/data/scratch/data/', transform=tforms.ToTensor())
        test_set = Imagenet64(train=False, root='/mnt/data/scratch/data/', transform=tforms.ToTensor())
    elif args.data == 'celebahq':
        im_dim = 3
        im_size = 256 if args.imagesize is None else args.imagesize
        train_set = CelebAHQ(
            train=True, root='/mnt/data/scratch/data/', transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor()
            ])
        )
        test_set = CelebAHQ(
            train=False, root='/mnt/data/scratch/data/',  transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor()
            ])
        )
    data_shape = (im_dim, im_size, im_size)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape




def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    model = odenvp.ODENVP(
        (args.batch_size, *data_shape),
        n_blocks=args.num_blocks,
        intermediate_dims=hidden_dims,
        div_samples=args.div_samples,
        strides=strides,
        squeeze_first=args.squeeze_first,
        nonlinearity=args.nonlinearity,
        layer_type=args.layer_type,
        zero_last=args.zero_last,
        alpha=args.alpha,
        cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
    )

    return model


if __name__ == "__main__":

    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)


    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)
    set_cnf_options(args, model)



    # restore parameters
    if args.chkpt is not None:
        checkpt = torch.load(args.chkpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    fixed_z = cvt(torch.randn(args.test_batch_size, *data_shape))


    chkdir = os.path.dirname(args.chkpt)
    tedf = pd.read_csv(os.path.join(chkdir,'test.csv'))
    trdf = pd.read_csv(os.path.join(chkdir,'training.csv'))
    wall_clock = trdf['wall'].to_numpy()[-1]
    itr = trdf['itr'].to_numpy()[-1]
    best_loss = tedf['bpd'].min()
    begin_epoch = int(tedf['epoch'].to_numpy()[-1]+1) # not exactly correct
    
    nvals = 2**args.nbits

    model.eval()
    with torch.no_grad():

        if args.validate:
            cleanbpd = 0.
            dirtybpd = 0.
            for i, (x, y) in enumerate(test_loader):

                xdirty = add_noise(cvt(255*x), nbits=args.nbits)
                xclean = shift(cvt(255*x), nbits=args.nbits)

                # Dirty
                # -----
                zero = torch.zeros(xdirty.shape[0], 1).to(xdirty)
                z, delta_logp, _ = model(xdirty, zero)  # run model forward

                logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
                logpx = logpz - delta_logp

                logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
                bits_per_dim = -(logpx_per_dim - np.log(nvals)) / np.log(2)
                dirtybpd = bits_per_dim.detach().cpu().item()/(i+1) + i/(i+1) * dirtybpd

                # Clean
                # -----
                zero = torch.zeros(xclean.shape[0], 1).to(xclean)
                z, delta_logp, _ = model(xclean, zero)  # run model forward

                logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
                logpx = logpz - delta_logp

                logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
                bits_per_dim = -(logpx_per_dim - np.log(nvals)) / np.log(2)
                cleanbpd = bits_per_dim.detach().cpu().item()/(i+1) + i/(i+1) * cleanbpd
                print('[%3d/%3d] clean %5.3g, dirty %5.3g   '%(i, len(test_loader), cleanbpd, dirtybpd), end='\r')

            print('\nClean test bpd: %.5g'%cleanbpd)
            print('Dirty test bpd: %.5g'%dirtybpd)

        if args.save_real:
            for i, (x, y) in enumerate(test_loader):
                if i<10:
                    pass
                elif i==10:
                    real= x.size(0)
                else:
                    break
            fig_filename = os.path.join(chkdir, "real.jpg")
            save_image(x, fig_filename, nrow=args.nrow)

            

            



        if args.generate:
            print('\nGenerating images... ')
            for t in [1.,0.9,0.8,0.7,0.6,0.5]:
                # visualize samples and density
                fig_filename = os.path.join(chkdir, "generated-T%g.jpg"%t)
                utils.makedirs(os.path.dirname(fig_filename))
                generated_samples = model(t*fixed_z, reverse=True)
                x = unshift(generated_samples[0].view(-1, *data_shape), 8)
                save_image(x, fig_filename, nrow=args.nrow)
