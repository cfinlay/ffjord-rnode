import torch
import torch.nn as nn
import lib.layers as layers
from lib.layers.odefunc import ODEnet
from lib.layers.squeeze import squeeze, unsqueeze
import numpy as np


class ODENVP(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(
            self,
            input_size,
            n_scale=float('inf'),
            n_blocks=2,
            strides=None,
            intermediate_dims=(32,),
            nonlinearity="softplus",
            layer_type="concat",
            squash_input=True,
            squeeze_first=False,
            zero_last=True,
            div_samples=1,
            alpha=0.05,
            cnf_kwargs=None,
            training_complete_model=False,
            training_last_layer=False
    ):
        super(ODENVP, self).__init__()
        if squeeze_first:
            bsz, c, w, h = input_size
            c, w, h = c * 4, w // 2, h // 2
            input_size = bsz, c, w, h
        
        if training_complete_model:    
            self.n_scale = min(n_scale, self._calc_n_scale(input_size))
        else:
            self.n_scale = 1
    
        self.n_blocks = n_blocks
        self.intermediate_dims = intermediate_dims
        self.layer_type = layer_type
        self.zero_last = zero_last
        self.div_samples = div_samples
        self.nonlinearity = nonlinearity
        self.strides = strides
        self.squash_input = squash_input
        self.alpha = alpha
        self.squeeze_first = squeeze_first
        self.cnf_kwargs = cnf_kwargs if cnf_kwargs else {}
        self.use_mse = not(training_complete_model | training_last_layer)
        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)
        
        if training_complete_model:
            self.transforms = self.build_net_for_complete(input_size)
        else:
        
            self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

    def build_net_for_complete(self, input_size):
        batch, c,h,w = input_size
        models = []
        
            
        model = self._build_net((batch, c, h, w))
        final_state_dict = {}
        skip_index = len("transforms.")
        for i in range(self.n_scale):
            
            state_dict = torch.load("/HPS/CNF/work/ffjord-rnode/experiments/celebahq/example/best_"+ str(i)+".pth")["state_dict"]
            
            for key in state_dict.keys():
                final_state_dict[str(i) + key[skip_index+1:]]= state_dict[key]
        
        model_dict_keys = list(model.state_dict().keys())
        modified_state_dict ={}
        for idx,key in enumerate(final_state_dict.keys()):
            modified_state_dict[model_dict_keys[idx]] = final_state_dict[key]
        
        model.load_state_dict(modified_state_dict)
        return model
            
    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    div_samples=self.div_samples,
                    zero_last=self.zero_last,
                    layer_type=self.layer_type,
                    strides=self.strides,
                    idims=self.intermediate_dims,
                    squeeze=True,  # don't squeeze last layer
                    init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform())
                    if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i <= self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            out = self._generate(x, logpx, reg_states)
            if self.squeeze_first:
                x = unsqueeze(out[0])
            else:
                x = out[0]
            return x, out[1], out[2]
        else:
            if self.squeeze_first:
                x = squeeze(x)
            if self.use_mse:
                return self._logdensity_single(x, logpx, reg_states)
            else:
                return self._logdensity(x, logpx, reg_states)

    def _logdensity(self, x, logpx=None, reg_states=tuple()):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        for idx in range(len(self.transforms)):
            x, _logpx, reg_states = self.transforms[idx].forward(x, _logpx, reg_states)
            if idx < len(self.transforms) - 1:
                d = x.size(1) // 2
                x, factor_out = x[:, :d], x[:, d:]
            else:
                # last layer, no factor out
                factor_out = x
            out.append(factor_out)
        out = [o.view(o.size()[0], -1) for o in out]
        out = torch.cat(out, 1)
        return out, _logpx, reg_states

    def _logdensity_single(self, x, logpx=None, reg_states=tuple()):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        _, dim, im_size, _ = x.shape
        out = []
        for idx in range(len(self.transforms)):
            x, _logpx, reg_states = self.transforms[idx].forward(x, _logpx, reg_states)
            if idx <= len(self.transforms) - 1:
                d = x.size(1) // 2
                x, factor_out = x[:, :d], x[:, d:]
            else:
                # last layer, no factor out
                factor_out = x
            out.append(factor_out)
        out = [o.view(o.size()[0], -1) for o in out]
        out = torch.cat(out, 1)
        return out, _logpx, reg_states, x

    def _generate(self, z, logpz=None, reg_states=tuple()):
        z = z.view(z.shape[0], -1)
        zs = []
        i = 0
        for dims in self.dims:
            s = np.prod(dims)
            zs.append(z[:, i:i + s])
            i += s
        zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        z_prev, _logpz, _ = self.transforms[-1](zs[-1], _logpz, reverse=True)
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            z_prev, _logpz, reg_states = self.transforms[idx](z_prev, _logpz, reg_states, reverse=True)
        return z_prev, _logpz, reg_states


class StackedCNFLayers(layers.SequentialFlow):
    def __init__(
            self,
            initial_size,
            idims=(32,),
            nonlinearity="softplus",
            layer_type="concat",
            div_samples=1,
            squeeze=True,
            init_layer=None,
            n_blocks=1,
            zero_last=True,
            strides=None,
            cnf_kwargs={},
    ):
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size):
            net = ODEnet(idims, size, strides, True, layer_type=layer_type, nonlinearity=nonlinearity,
                         zero_last_weight=zero_last)
            f = layers.ODEfunc(net, div_samples=div_samples)
            return f

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [layers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]
            post = [layers.CNF(_make_odefunc(after_squeeze_size), **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [layers.SqueezeLayer(2)] + post
        else:
            chain += [layers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]

        super(StackedCNFLayers, self).__init__(chain)
