import os
import yaml
import torch
from typing import Optional
from src import utils
from ..utils.data_utils import OrderReader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

log = utils.get_logger(__name__)


class LabelDataModule(LightningDataModule):
    """LightningDataModule for Weather dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        prepared_folder,
        look_back,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.prepared_folder = prepared_folder
        self.look_back = look_back
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def append_to_yaml(file_path, data_to_append):
        with open(file_path, 'a') as file:
            yaml.dump(data_to_append, file)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        self.train_dataset = OrderReader(self.prepared_folder, self.look_back, 'train')
        self.valid_dataset = OrderReader(self.prepared_folder, self.look_back, 'valid')
        self.test_dataset = OrderReader(self.prepared_folder, self.look_back, 'test')
        self.cat_vocab_size = self.train_dataset.cat_vocab_size
        self.id_vocab_size = self.train_dataset.id_vocab_size
        self.amount_vocab_size = self.train_dataset.amount_vocab_size
        self.dt_vocab_size = self.train_dataset.dt_vocab_size
        self.max_cat_len = self.train_dataset.max_cat_len
        print(self.cat_vocab_size, self.id_vocab_size, self.amount_vocab_size, self.dt_vocab_size, self.max_cat_len)
        # file_path = '/NOTEBOOK/hermes/nir/lanet_lightning/configs/model/lanet.yaml'
        # data_to_append = {'cat_vocab_size': self.cat_vocab_size,
        #                   'id_vocab_size': self.id_vocab_size,
        #                   'amount_vocab_size': self.amount_vocab_size,
        #                   'dt_vocab_size': self.dt_vocab_size,
        #                   'max_cat_len': self.max_cat_len}
        # LabelDataModule.append_to_yaml(file_path, data_to_append)
        # print(self.cat_vocab_size, self.id_vocab_size, self.amount_vocab_size, self.dt_vocab_size, self.max_cat_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
