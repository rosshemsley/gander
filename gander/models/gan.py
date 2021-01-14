"""
TODO(Ross): Ideas for improvement
* label smoothing
* add semantic labels! (conditional GAN)
* add std deviation computation to create more diversity
* Implement correct model state loading
"""

from typing import Tuple
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision

from gander.datasets import CelebA, denormalize
from torch.utils.data import DataLoader

from .stage_manager import Stage
from .modules import Descriminator, Generator, resample


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
        return DataLoader(
            self.dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
        )

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

        z = self.random_latent_vectors(batch_size).type_as(x_r)
        x_g = self.generator(z, layers, alpha)

        x_hat = _random_sample_line_segment(x_r, x_g)
        gp = _gradient_penalty(x_hat, self.descriminator, layers, alpha)

        f_r = self.descriminator(x_r, layers, alpha)
        f_g = self.descriminator(x_g, layers, alpha)

        wgan_loss = f_g.mean() - f_r.mean()
        gp_loss = gp.mean()

        loss = wgan_loss + 10 * gp_loss

        self.log("wgan loss", wgan_loss)
        self.log("gp loss", gp_loss)
        self.log("descriminator_loss", loss)

        if batch_idx % self.stage.batches_between_image_log == 0:
            grid_g = torchvision.utils.make_grid(denormalize(x_g[0:20]), nrow=4)
            grid_r = torchvision.utils.make_grid(denormalize(x_r[0:20]), nrow=4)
            self.logger.experiment.add_image(
                "images.generated", grid_g, self.total_steps_taken
            )
            self.logger.experiment.add_image(
                "images.train", grid_r, self.total_steps_taken
            )

        return loss

    def generator_step(self, x, batch_idx):
        layers = self.stage.num_layers
        alpha = self.stage.progress
        batch_size = x.size(0)

        latent_vectors = self.random_latent_vectors(batch_size).type_as(x)

        y = self.generator(latent_vectors, layers, alpha)
        f = self.descriminator(y, layers, alpha)
        loss = -f.mean()

        self.log("generator_loss", loss)
        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        """
        We alternate between using the batch, and ignoring it.
        Technically, this is inefficient, since we always pay the cost of loading the data.
        """
        alpha = self.stage.progress
        layers = self.stage.num_layers
        batch_size = self.stage.batch_size

        x, _ = batch
        x = _soft_resample(x, alpha, _image_resolution(self.conf, layers))

        self.log("stage.progress", alpha * 100, prog_bar=True)
        self.log("stage.percent_complete", alpha * 100)
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


def _random_sample_line_segment(x1, x2):
    """
    Given two tensors [B,C,H,W] of equal dimensions, in a batch of size B.
    Return a tensor containing B samples randomly sampled on the line segment between each point x1[i], x2[i].
    """
    batch_size = x1.size(0)
    epsilon = _unif(batch_size).type_as(x1)
    return epsilon[:, None, None, None] * x1 + (1 - epsilon)[:, None, None, None] * x2


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
    gp = (gradient_norm - 1.0) ** 2

    # We must zero the gradient accumulators of the network parameters, to avoid
    # the gradient computation of x in this function affecting the backwards pass of the optimizer.
    x.grad = None
    descriminator.zero_grad()

    return gp


def _unif(batch_size):
    return torch.distributions.uniform.Uniform(0, 1).sample([batch_size])


def _soft_resample(
    x: torch.Tensor, alpha: float, resolution: Tuple[int, int]
) -> torch.Tensor:
    """
    Softly resample the image from half the size to the full size.
    """
    r1 = resolution
    r2 = resolution[0] // 2, resolution[1] // 2
    x1 = resample(resample(x, r2), r1)
    x2 = resample(x, r1)

    return (1 - alpha) * x1 + alpha * x2
