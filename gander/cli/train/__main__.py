import pathlib
import click

from omegaconf import OmegaConf
import pytorch_lightning as pl
from gander.models import GAN, StageManager


@click.command()
@click.option(
    "--root-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Root directory of the interaction dataset.",
)
def main(root_dir: str):
    # floob()
    # return

    conf = OmegaConf.load("config.yaml")
    conf.root_dir = root_dir

    print(OmegaConf.to_yaml(conf))

    model = GAN(conf)
    stage_manager = StageManager(conf)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100000,
        # precision=16,
        # gradient_clip_val=0.5,
        callbacks=[stage_manager],
        reload_dataloaders_every_epoch=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn
# class MyMod(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.theta =


def floob():
    import torch

    def f(x, theta):
        return torch.pow((x), 2) * theta

    x = torch.Tensor([3])
    theta = torch.Tensor([5])

    x.requires_grad = True
    theta.requires_grad = True

    y = f(x, theta)

    y.backward(create_graph=True)
    # torch.autograd.backward(y, create_graph=True)
    print("autograd x grad", x.grad)

    z = y + x.grad
    # z = x.grad

    print("theta grad", theta.grad)
    print("z evaluated", z)
    theta.grad = None
    z.backward()

    print("theta grad", theta.grad)

    # print ("x grad", x.grad)
    # print ("theta grad", theta.grad)
    # # print ("y grad", y.grad)

    # print("backwards again...")

    # x.grad = None
    # theta.grad = None
    # y.backward(retain_graph=True)

    # print ("x grad", x.grad)
    # print ("theta grad", theta.grad)
