import os
from lightning.pytorch.callbacks import Callback


class ModelCheckpointCallback(Callback):
    def __init__(self, save_dir: str, every_n_epochs: int = 1):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        epoch = trainer.current_epoch
        print(f"\n[Epoch {epoch}] Saving model checkpoint...")

        checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        for param in pl_module.model.parameters():  # type: ignore
            param.data = param.data.contiguous()

        for buffer in pl_module.model.buffers():  # type: ignore
            buffer.data = buffer.data.contiguous()

        pl_module.model.save_pretrained(checkpoint_dir)  # type: ignore
        pl_module.processor.save_pretrained(checkpoint_dir)  # type: ignore

        latest_symlink = os.path.join(self.save_dir, "latest")
        if os.path.exists(latest_symlink) or os.path.islink(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(f"epoch_{epoch}", latest_symlink)

        print(f"Checkpoint saved to: {checkpoint_dir}")
        print(f"Symlink 'latest' points to: {latest_symlink}\n")
