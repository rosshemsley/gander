import pathlib
import click

from omegaconf import OmegaConf
import pytorch_lightning as pl
from gander.models import GAN

@click.command()
@click.option(
    "--root-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Root directory of the interaction dataset.",
)
def main(root_dir: str):
    conf = OmegaConf.load("config.yaml")
    conf.root_dir = root_dir

    print(OmegaConf.to_yaml(conf))

    model = GAN(conf)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100000,
        gradient_clip_val=0.5,
        precision=16,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
