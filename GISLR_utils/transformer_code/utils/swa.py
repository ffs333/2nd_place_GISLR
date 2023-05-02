from copy import deepcopy
from typing import Optional, Union, Dict, Any

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only


class SWA(pl.Callback):
    """Implements swa (exponential moving average) to any kind of model.
    swa weights will be used during validation and stored separately from original model weights.

    How to use swa:
        - Sometimes, last swa checkpoint isn't the best as swa weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See swa in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast swa weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing swa weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """
    def __init__(self, swa_epoch: int = 5, swa_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.swa_epoch = swa_epoch
        self.swa_device: str = f"{swa_device}" if swa_device else None  # perform swa on different device from the model
        self.swa_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.swa_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self.swa_count = 0

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.
        
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()
        
    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of swa weights in rank zero.
        if self.swa_count == 0 and pl_module.global_rank == 0 and (trainer.max_epochs - pl_module.current_epoch) < self.swa_epoch:
            self.swa_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.swa_device:
                self.swa_state_dict = {k: tensor.to(device=self.swa_device) for k, tensor in self.swa_state_dict.items()}

            if self.swa_device == "cpu" and self.swa_pin_memory:
                self.swa_state_dict = {k: tensor.pin_memory() for k, tensor in self.swa_state_dict.items()}

            self.swa_count = 1

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update swa weights
        if (trainer.max_epochs - pl_module.current_epoch) < self.swa_epoch:
            with torch.no_grad():
                for key, value in self.get_state_dict(pl_module).items():
                    swa_value = self.swa_state_dict[key]
                    swa_value.copy_((swa_value * self.swa_count + value) / (self.swa_count + 1), non_blocking=True)

            self.swa_count += 1

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.swa_count == 0:
            return  # Skip Lightning sanity validation check if no swa weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        pl_module.trainer.strategy.broadcast(self.swa_state_dict, 0)
        assert self.swa_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the swa static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.swa_state_dict.keys()}"
        pl_module.load_state_dict(self.swa_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove swa state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.swa_state_dict = {}

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.swa_count == 0:
            return  # Skip Lightning sanity validation check if no swa weights has been loaded from a checkpoint.

        # Replace swa weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> dict:
        return {"swa_state_dict": self.swa_state_dict, "swa_count": self.swa_count}

    @overrides
    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        self.swa_count = callback_state["swa_count"]
        self.swa_state_dict = callback_state["swa_state_dict"]