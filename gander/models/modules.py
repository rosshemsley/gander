from typing import Tuple

import torch.nn as nn
import torch


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

        self.first_layer = nn.Linear(
            latent_dims, first_layer_size[0] * first_layer_size[1] * channels
        )

        self.layers = nn.ModuleList(
            [
                Layer(channels, channels, num_groups)
                for _ in range(conf.model.num_layers)
            ]
        )

        self.to_rgb = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            _conv(in_channels=channels, out_channels=3, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, num_layers, weight) -> torch.Tensor:
        x = self.first_layer(x)
        x = x.reshape(
            -1, self.conf.model.conv_channels, *self.conf.model.first_layer_size
        )

        if num_layers == 0:
            return self.to_rgb(x)

        for l in self.layers[: num_layers - 1]:
            x = _double_resolution(x)
            x = l(x)

        prev_image = _double_resolution(self.to_rgb(x))

        l = self.layers[num_layers - 1]
        x = _double_resolution(x)
        new_image = self.to_rgb(l(x))

        return (weight) * new_image + (1 - weight) * prev_image


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

        self.layers = nn.ModuleList(
            [
                Layer(channels, channels, num_groups)
                for _ in range(conf.model.num_layers)
            ]
        )

        self.critic = Critic(conf)

        # clip_value = 0.01
        # for p in self.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def forward(self, x: torch.Tensor, num_layers, weight) -> torch.Tensor:
        prev_x = self.from_rgb(_half_resolution(x))
        x = self.from_rgb(x)

        if num_layers == 0:
            return self.critic(x)

        l = self.layers[num_layers - 1]
        x = (weight) * _half_resolution(l(x)) + (1 - weight) * prev_x

        for l in reversed(self.layers[0 : num_layers - 1]):
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
        return self.critic(x)


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            _conv(in_channels, out_channels),
        )

    def forward(self, x):
        return nn.LeakyReLU()(x + self.conv(x))


def resample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # TODO(Ross): check align_corners and interpolation mode.
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


# TODO(Ross): think about bias.
def _conv(
    in_channels: int, out_channels: int, kernel_size=3, padding=1, stride=1, bias=False
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
    )


def _double_resolution(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    return resample(x, size=(h * 2, w * 2))


def _half_resolution(x: torch.Tensor) -> torch.Tensor:
    return nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x)
