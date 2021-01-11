"""
Ideas:
* Increase fc layers in classifier
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

        self.classifier = Classifier(conf)

    def forward(self, x: torch.Tensor, num_layers, weight) -> torch.Tensor:
        prev_x = self.from_rgb(_half_resolution(x))
        x = self.from_rgb(x)

        if num_layers == 0:
            return self.classifier(x)

        l = self.layers[num_layers-1]
        x = (weight) * _half_resolution(l(x)) + (1-weight) * prev_x

        for l in reversed(self.layers[0:num_layers-1]):
            x = l(x)
            x = _half_resolution(x)

        return self.classifier(x)


class Classifier(nn.Module):
    def __init__(self, conf):
        super().__init__()

        resolution = conf.model.first_layer_size
        channels = conf.model.conv_channels
        fc_layers = conf.model.fc_layers

        self.classifier = nn.Sequential(
            nn.Linear(resolution[0] * resolution[1] * channels, fc_layers),
            nn.LeakyReLU(),
            nn.Linear(fc_layers, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.classifier(x).clamp(min=1e-5, max = 1 - 1e-5)


class GAN(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.generator = Generator(conf)
        self.descriminator = Descriminator(conf)

        self.stage = None
    
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

    def descriminator_step(self, batch, batch_idx):
        layers = self.stage.num_layers
        alpha = self.stage.progress

        x = _resample(batch, _image_resolution(self.conf, layers))

        grid = torchvision.utils.make_grid(denormalize(x[0:16]), nrow=4)
        if batch_idx % self.stage.batches_between_image_log == 0:
            self.logger.experiment.add_image("train images", grid, DES_STEP_COUNT)

        batch_size = x.size(0)

        latent_vectors = self.random_latent_vectors(batch_size).type_as(x)
        y = self.generator(latent_vectors, layers, alpha)
        p_gen = self.descriminator(y, layers, alpha)

        desciminator_correct_fake = (p_gen < 0.5).sum()
        nll_generated = -torch.log(1 - p_gen)

        fake_percent_correct = (desciminator_correct_fake / batch_size) * 100
        self.log("fake_percent_correct", fake_percent_correct)

        p_real = self.descriminator(x, layers, alpha)
        desciminator_correct_real = (p_real > 0.5).sum()
        nll_real = - torch.log(p_real)

        real_percent_correct = (desciminator_correct_real / batch_size) * 100
        self.log("real_percent_correct", real_percent_correct)

        loss = torch.vstack([nll_generated, nll_real]).mean()
        self.log("descriminator_loss", loss)
        self.log("success rate", 100*(desciminator_correct_fake + desciminator_correct_real) / (2*batch_size))

        return loss

    def generator_step(self, batch, batch_idx):
        layers = self.stage.num_layers
        alpha = self.stage.progress

        batch_size = batch.size(0)
        latent_vectors = self.random_latent_vectors(batch_size).type_as(batch)

        y = self.generator(latent_vectors, layers, alpha)
        p = self.descriminator(y, layers, alpha)
        loss = -torch.log(p).mean()

        grid = torchvision.utils.make_grid(denormalize(y[0:16]), nrow=4)
        if batch_idx % self.stage.batches_between_image_log == 0:
            self.logger.experiment.add_image("generated images", grid, GEN_STEP_COUNT)

        self.log("generator_loss", loss)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int):
        """
        We alternate between using the batch, and ignoring it.
        Technically, this is inefficient, since we always pay the cost of loading the data.
        """
        x, _ = batch

        alpha = self.stage.progress
        layers = self.stage.num_layers
        batch_size = self.stage.batch_size

        self.log("stage percent complete", alpha*100)
        self.log("num_layers", layers)
        self.log("batch_size", batch_size)

        x = _soft_resample(x, alpha, _image_resolution(self.conf, layers))

        if optimizer_idx == 0:
            global GEN_STEP_COUNT
            GEN_STEP_COUNT += 1
            return self.generator_step(x, batch_idx)
        else:
            global DES_STEP_COUNT
            DES_STEP_COUNT += 1
            return self.descriminator_step(x, batch_idx)

    def configure_optimizers(self):
        lr = self.conf.trainer.learning_rate
        return [
            torch.optim.Adam(self.generator.parameters(), lr=lr),
            torch.optim.Adam(self.descriminator.parameters(), lr=lr),
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


def _resample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # TODO(Ross): check align_corners and interpolation mode.
    return nn.functional.interpolate(x, size=size, mode="bilinear")


def _double_resolution(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    return _resample(x, size=(h * 2, w * 2))


def _half_resolution(x: torch.Tensor) -> torch.Tensor:
    return nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x)


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

def _interpolate_factors(alpha, n):
    if n == 1:
        return [1.0]

    x = alpha * 1/n
    remainder = 1-x

    result = []
    for i in range(n-1):
        v = (1-x) * (1/(n-1))
        result.append(v)

    result.append(x)

    return result

def _soft_resample(x, alpha, resolution):
    """
    Softly resample the image from half the size to the full size.
    """

    r1 = resolution
    r2 = resolution[0] // 2, resolution[1] // 2

    x1 = _resample(_resample(x, r2), r1)
    x2 = _resample(x, r1)

    return (1-alpha) * x1 + alpha * x2
