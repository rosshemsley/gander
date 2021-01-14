"""
Ideas:
* label smoothing
* flip labels with some probability
* DCGAN?
* ADAM for generator 0.002! SGD for descriminator
* add semantic labels! (conditional GANs)
* Add noise to the inputs (both real and generated). Decay over time
* Train descriinator more?!


* Increase fc layers in critic
* add std deviation computation to create more diversity
* implement an evaluation function to support tuning
* Investigate different learning rates
* Try half precision
* Implement a way to look at gradient flow
* Implement correct model state loading
* remove batch norm on from_rgb layer
"""

from typing import Tuple
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision

from gander.datasets import CelebA, denormalize
from torch.utils.data import DataLoader

from .stage_manager import Stage

DES_STEP_COUNT = 0
GEN_STEP_COUNT = 0

class Generator(nn.Module):
    """
    Take a vector sampled from the latent space, and return a generated
    output from the net.
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        latent_dims = conf.model.latent_dims
        channels = conf.model.conv_channels
        num_groups = conf.model.num_groups
        first_layer_size = conf.model.first_layer_size

        self.first_layer = nn.Linear(latent_dims, first_layer_size[0] * first_layer_size[1] * channels)

        self.layers = nn.ModuleList([
            Layer(channels, channels, num_groups) for _ in range(conf.model.num_layers)
        ])

        self.to_rgb = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            _conv(in_channels=channels, out_channels=3, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, num_layers, weight) -> torch.Tensor:
        x = self.first_layer(x)
        x = x.reshape(-1, self.conf.model.conv_channels, *self.conf.model.first_layer_size)

        if num_layers == 0:
            return self.to_rgb(x)

        for l in self.layers[:num_layers-1]:
            x = _double_resolution(x)
            x = l(x)

        prev_image = _double_resolution(self.to_rgb(x))

        l = self.layers[num_layers-1]
        x = _double_resolution(x)
        new_image = self.to_rgb(l(x))

        return (weight) * new_image + (1-weight) * prev_image


class Descriminator(nn.Module):
    """
    Given a sample from the output of the generated distribution, encode the result
    back into 
    """
    def __init__(self, conf):
        super().__init__()
        channels = conf.model.conv_channels
        num_groups = conf.model.num_groups

        self.from_rgb = nn.Sequential(
            nn.BatchNorm2d(3),
            _conv(in_channels=3, out_channels=channels, kernel_size=1, padding=0),
            nn.LeakyReLU(),
        )

        self.layers = nn.ModuleList([
            Layer(channels, channels, num_groups) for _ in range(conf.model.num_layers)
        ])

        self.critic = Critic(conf)

        # clip_value = 0.01
        # for p in self.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))


    def forward(self, x: torch.Tensor, num_layers, weight) -> torch.Tensor:
        prev_x = self.from_rgb(_half_resolution(x))
        x = self.from_rgb(x)

        if num_layers == 0:
            return self.critic(x)

        l = self.layers[num_layers-1]
        x = (weight) * _half_resolution(l(x)) + (1-weight) * prev_x

        for l in reversed(self.layers[0:num_layers-1]):
            x = l(x)
            x = _half_resolution(x)

        return self.critic(x)


class Critic(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.epsilon = conf.model.min_confidence
        resolution = conf.model.first_layer_size
        channels = conf.model.conv_channels
        fc_layers = conf.model.fc_layers

        self.critic = nn.Sequential(
            nn.Linear(resolution[0] * resolution[1] * channels, fc_layers),
            nn.LeakyReLU(),
            nn.Linear(fc_layers, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.critic(x).clamp(min=self.epsilon)


class GAN(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.generator = Generator(conf)
        self.descriminator = Descriminator(conf)

        self.stage = None
        self.total_steps_taken = 0
    
    def set_current_stage(self, stage: Stage):
        print("Setting training stage", stage)
        self.stage = stage

    def prepare_data(self):
        self.dataset = CelebA(self.conf.root_dir)

    def random_latent_vectors(self, n: int):
        """
        Generate a random latent vector.
        The latent vector is distributed as N(0,1)^(latent_shape).
        """
        d = self.conf.model.latent_dims
        return torch.normal(torch.zeros(n, d), torch.ones(n, d))

    def train_dataloader(self, reload_dataloaders_every_epoch=True):
        batch_size = self.stage.batch_size
        num_workers = self.conf.trainer.num_dataloaders
        return DataLoader(self.dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    def forward(self, latent_vectors):
        # TODO
        ...

    def descriminator_step(self, x_r, batch_idx):
        """
        Takes a real sample from the data distribution, x_r, computes the critic loss.
        """
        layers = self.stage.num_layers
        alpha = self.stage.progress
        batch_size = x_r.size(0)

        if batch_idx % self.stage.batches_between_image_log == 0:
            grid = torchvision.utils.make_grid(denormalize(x_r[0:20]), nrow=4)
            self.logger.experiment.add_image("train images", grid, self.total_steps_taken)

        z = self.random_latent_vectors(batch_size).type_as(x_r)
        x_g = self.generator(z, layers, alpha)

        x_hat = _random_sample_line_segment(x_r, x_g)
        gp = _gradient_penalty(x_hat, self.descriminator, layers, alpha)

        f_r = self.descriminator(x_r, layers, alpha)
        f_g = self.descriminator(x_g, layers, alpha)    

        wgan_loss = f_g.mean() - f_r.mean()
        gp_loss = gp.mean()
    
        print("WGAN loss", wgan_loss)
        print("grad penalty", gp_loss)

        loss = wgan_loss + gp_loss

        self.log("descriminator_loss", loss)

        return loss

    def generator_step(self, x, batch_idx):
        layers = self.stage.num_layers
        alpha = self.stage.progress
        batch_size = x.size(0)

        latent_vectors = self.random_latent_vectors(batch_size).type_as(x)

        y = self.generator(latent_vectors, layers, alpha)
        f = self.descriminator(y, layers, alpha)
        loss = -f.mean()

        if batch_idx % self.stage.batches_between_image_log == 0:
            grid = torchvision.utils.make_grid(denormalize(y[0:20]), nrow=4)
            self.logger.experiment.add_image("generated images", grid, self.total_steps_taken)

        self.log("generator_loss", loss)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int):
        """
        We alternate between using the batch, and ignoring it.
        Technically, this is inefficient, since we always pay the cost of loading the data.
        """
        alpha = self.stage.progress
        layers = self.stage.num_layers
        batch_size = self.stage.batch_size

        x, _ = batch
        x = _resample(x, _image_resolution(self.conf, layers))
        # x = _soft_resample(x, alpha, _image_resolution(self.conf, layers))

        self.log("stage.progress", alpha*100, prog_bar=True)
        self.log("stage.percent_complete", alpha*100)
        self.log("num_layers", layers)
        self.log("batch_size", batch_size)

        if optimizer_idx == 0:
            self.total_steps_taken += 1
            return self.descriminator_step(x, batch_idx)
        else:
            if batch_idx % 5 == 0:
                return self.generator_step(x, batch_idx)
            else:
                return None

    def configure_optimizers(self):
        lr = self.conf.trainer.learning_rate
        adam_epsilon = self.conf.trainer.epsilon
        return [
            # RMSprop since the critic tends not to be stationary (see WGAN paper)
            # torch.optim.RMSprop(self.descriminator.parameters(), lr=0.001),
            torch.optim.Adam(self.descriminator.parameters(), lr=lr, eps=adam_epsilon),
            torch.optim.Adam(self.generator.parameters(), lr=lr, eps=adam_epsilon),
        ]


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            _conv(in_channels, out_channels),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        return x + self.conv(x)


# TODO(Ross): think about bias.
def _conv(in_channels: int, out_channels: int, kernel_size=3, padding=1, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)


def _double_resolution(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    return _resample(x, size=(h * 2, w * 2))


def _half_resolution(x: torch.Tensor) -> torch.Tensor:
    return nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x)


def _resample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # TODO(Ross): check align_corners and interpolation mode.
    return nn.functional.interpolate(x, size=size, mode="bilinear")


def _image_resolution(conf, max_layers=None) -> Tuple[int, int]:
    """
    The resolution of the laster layer of the generator network
    """
    h, w = conf.model.first_layer_size
    n = conf.model.num_layers
    if max_layers is not None:
        n = min(max_layers, n)
    factor = pow(2, n)
    return h * factor, w * factor


def _soft_resample(x, alpha, resolution):
    """
    Softly resample the image from half the size to the full size.
    """

    r1 = resolution
    r2 = resolution[0] // 2, resolution[1] // 2

    x1 = _resample(_resample(x, r2), r1)
    x2 = _resample(x, r1)

    return (1-alpha) * x1 + alpha * x2


def _random_sample_line_segment(x1, x2):
    """
    Given two tensors [B,C,H,W] of equal dimensions, in a batch of size B.
    Return a tensor containing B samples randomly sampled on the line segment between each point x1[i], x2[i].
    """
    batch_size = x1.size(0)
    epsilon = _unif(batch_size).type_as(x1)
    return epsilon[:, None, None, None] * x1 + (1-epsilon)[:, None, None, None] * x2


def _gradient_penalty(x, descriminator, layers, alpha):
    """
    Compute the gradient penalty term used in WGAN-gp.
    Returns the gradient penalty for each batch entry, the loss term is computed as the average.

    Works by sampling points on the line segment between x_r and x_g, then computing the gradient
    of the critic with respect to each sample point.
    """
    batch_size = x.size(0)

    # We compute the gradient of the parameters using the regular autograd.
    # The key to making this work is including `create_graph`, this means that the computations
    # in this penalty will be added to the computation graph for the loss function, so that the
    # second partial derivatives will be correctly computed. 
    x.requires_grad = True
    f_x = descriminator(x, layers, alpha)
    f_x.backward(torch.ones_like(f_x), create_graph=True)

    grad_x_flat = x.grad.view(batch_size, -1)
    gradient_norm = torch.linalg.norm(grad_x_flat, dim=1)
    gp = (gradient_norm - 1.0)**2

    # We must zero the gradient accumulators of the network parameters, to avoid
    # the gradient computation of x in this function affecting the backwards pass of the optimizer.
    x.grad = None
    descriminator.zero_grad()

    return gp


def _unif(batch_size):
    return torch.distributions.uniform.Uniform(0,1).sample([batch_size])
