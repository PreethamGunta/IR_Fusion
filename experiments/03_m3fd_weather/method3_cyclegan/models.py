"""CycleGAN networks, verbatim from the reference implementation.

Every layer here is a deliberate copy of the original junyanz/pytorch-CycleGAN
definitions rather than a paraphrase, for one concrete reason: the pretrained
weights we may want to initialise from -- clear2foggy.pth / clear2rainy.pth /
clear2snowy.pth from hgupta01/Weather_Effect_Generator -- were produced against
exactly these modules.  lib/gan_networks.py in that repository was checked and
is this architecture: ResnetGenerator(3, 3, ngf=64, n_blocks=9) with
InstanceNorm and reflect padding, NLayerDiscriminator(3, 64, 3) PatchGAN.

A paraphrase would change parameter names or ordering and the pretrained
weights would silently load as a near-empty state_dict.  So: do not
"improve" this file.

The one thing that is *not* fixed is the normalisation type.  CycleGAN's own
convention is instance norm, but hgupta01's define_G() defaults to norm='batch'.
Those two produce different state_dict keys (BatchNorm carries running_mean and
running_var, InstanceNorm does not), so whichever they trained with determines
whether the weights load.  train_cyclegan.py probes for the right one; see
load_pretrained there.
"""

import functools

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: nn.Identity()
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net, init_type="normal", init_gain=0.02):
    """Original pix2pix/CycleGAN initialisation."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                 use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Fully convolutional. That is the whole reason we can train on 512x512
    crops and apply the result at M3FD's native 1024x768: there is no
    fully-connected layer and no fixed input size anywhere in this module."""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, padding_type="reflect"):
        assert n_blocks >= 0
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------------
# discriminator
# --------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                               kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------

class GANLoss(nn.Module):
    """LSGAN. The last discriminator layer has no sigmoid, so the loss is
    applied directly to the logits."""

    def __init__(self, gan_mode="lsgan", target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] is not implemented")

    def get_target_tensor(self, prediction, target_is_real):
        return (self.real_label if target_is_real else self.fake_label).expand_as(prediction)

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ("lsgan", "vanilla"):
            return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))
        if self.gan_mode == "wgangp":
            return -prediction.mean() if target_is_real else prediction.mean()
        raise NotImplementedError(f"GAN mode [{self.gan_mode}] is not implemented")


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def define_G(input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm="instance",
             init_type="normal", init_gain=0.02):
    net = ResnetGenerator(input_nc, output_nc, ngf,
                          norm_layer=get_norm_layer(norm),
                          use_dropout=False, n_blocks=n_blocks)
    return init_weights(net, init_type, init_gain)


def define_D(input_nc=3, ndf=64, n_layers=3, norm="instance",
             init_type="normal", init_gain=0.02):
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers,
                              norm_layer=get_norm_layer(norm))
    return init_weights(net, init_type, init_gain)


class ImagePool:
    """Shrivastava-style buffer of previously generated images.

    Without it the discriminator sees only the generator's current output and
    overfits to it within a few iterations, which shows up as the classic
    oscillation of GAN loss.
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out.append(img)
            elif torch.rand(1).item() > 0.5:
                idx = torch.randint(0, self.pool_size, (1,)).item()
                out.append(self.images[idx].clone())
                self.images[idx] = img
            else:
                out.append(img)
        return torch.cat(out, 0)
