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
    StageManager tracks current "stage" of training.
    The gander package trains in several stages, each with an increasing number of layers.
    This class manages this state for the model.
    When checkpointing the model, this information is stored inside of it.
    """

    def __init__(self):
        self.index = 0
        self.current_epoch = 0
        self.current_step = 0

    def on_train_start(self, trainer, module):
        self._update_stage(module.conf)
        module.set_current_stage(self.current_stage)

    def on_epoch_end(self, trainer, module):
        self.current_epoch += 1
        if self.current_epoch == self.current_stage.epochs:
            path = os.path.join(
                module.logger.log_dir, "checkpoints", f"stage_{self.index}.ckpt"
            )
            print(f"Saving checkpoint to '{path}'...")

            self.current_epoch = 0
            self.current_step = 0
            self.index = min(self.index + 1, len(module.conf.training_stages) - 1)
            self._update_stage(module.conf)
            module.set_current_stage(self.current_stage)
            trainer.save_checkpoint(path)

    def on_batch_end(self, trainer, module):
        self.current_stage.progress = self.current_step / (
            self.current_stage.epochs * trainer.num_training_batches
        )
        self.current_step += 1

    def on_save_checkpoint(self, _, __):
        return {
            "index": self.index,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
        }

    def on_load_checkpoint(self, dct):
        self.index = dct["index"]
        self.current_epoch = dct["current_epoch"]
        self.current_step = 0

    def _update_stage(self, conf):
        cfg = conf.training_stages[self.index]
        stage = Stage(progress=0, **cfg)
        self.current_stage = stage
