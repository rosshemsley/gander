import pathlib
import click
from typing import Optional

from omegaconf import OmegaConf
import pytorch_lightning as pl
from gander.models import GAN, StageManager


@click.command()
@click.option(
    "--root-dir",
    required=False,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Root directory of the interaction dataset.",
)
@click.option(
    "--resume-checkpoint",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Given a checkpoint, resume training from that point.",
)
def main(root_dir: Optional[str], resume_checkpoint: Optional[str]):
    if resume_checkpoint is not None:
        print("Resuming training from checkpoint...")
        model = GAN.load_from_checkpoint(resume_checkpoint)
    else:
        conf = OmegaConf.load("config.yaml")
        model = GAN(conf)

    if root_dir is not None:
        model.conf.root_dir = root_dir

    print(OmegaConf.to_yaml(model.conf))

    stage_manager = StageManager()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100000,
        callbacks=[stage_manager],
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=resume_checkpoint,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
