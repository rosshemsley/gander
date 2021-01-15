import os

import pytorch_lightning as pl
from dataclasses import dataclass


@dataclass
class Stage:
    epochs: int
    batch_size: int
    num_layers: int
    progress: float
    batches_between_image_log: int


class StageManager(pl.Callback):
    """
    StageManager tracks the new concept of "stage"
    which is a group of epochs.
    """

    def __init__(self, conf):
        self.conf = conf
        self.index = 0
        self.current_epoch = 0
        self.current_step = 0

    def setup(self, trainer, module, _):
        cfg = self.conf.training_stages[self.index]
        stage = Stage(
            progress=0,
            **cfg,
        )
        self.current_stage = stage
        module.set_current_stage(stage)

    def on_epoch_end(self, trainer, module):
        self.current_epoch += 1
        if self.current_epoch == self.current_stage.epochs:
            path = os.path.join(
                module.logger.log_dir, "checkpoints", f"stage_{self.index}.ckpt"
            )
            print(f"saving checkpoint to '{path}'...")
            trainer.save_checkpoint(path)

            self.current_epoch = 0
            self.current_step = 0
            self.index = min(self.index + 1, len(self.conf.training_stages) - 1)
            cfg = self.conf.training_stages[self.index]
            stage = Stage(progress=0, **cfg)
            self.current_stage = stage
            module.set_current_stage(stage)

    def on_batch_end(self, trainer, module):
        self.current_stage.progress = self.current_step / (
            self.current_stage.epochs * trainer.num_training_batches
        )
        self.current_step += 1

    def on_save_checkpoint(_, __):
        return {
            "conf": self.conf,
            "index": self.index,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
        }

    def on_load_checkpoint(dct):
        self.conf = dct["conf"]
        self.index = dct["index"]
        self.current_epoch = dct["current_epoch"]
        self.current_step = dct["current_step"]
