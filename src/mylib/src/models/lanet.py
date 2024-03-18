from typing import Any, List
import sys
sys.path.append('/NOTEBOOK/hermes/nir/lanet_lightning/')
print(sys.path)

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn.functional import one_hot
from ..utils.metrics import valid_thr, calculate_all_metrics, multilabel_crossentropy_loss
from .compoments.encoder import TransformerEncoder

class LANET(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        look_back,
        cat_vocab_size,
        id_vocab_size,
        amount_vocab_size,
        dt_vocab_size,
        max_cat_len,
        emb_dim,
        lr: float = 0.003,
        weight_decay: float = 0.0,
    ):
        super(self.__class__, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.cat_vocab_size = cat_vocab_size
        self.id_vocab_size = id_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.dt_vocab_size = dt_vocab_size
        self.max_cat_len = max_cat_len

        self.look_back = look_back
        self.emb_dim = emb_dim

        self.weight_decay = weight_decay
        self.lr = lr

        self.transformer_encoder = TransformerEncoder(self.look_back, self.cat_vocab_size, self.id_vocab_size,
                                                      self.amount_vocab_size,
                                                      self.dt_vocab_size, self.emb_dim)

        self.dropout = nn.Dropout(0.3)
        self.encoder_history = nn.Linear(3 * emb_dim, 1)
        self.bn_history = nn.BatchNorm1d(cat_vocab_size)
        self.bn_labels = nn.BatchNorm1d(cat_vocab_size)
        self.relu = nn.ReLU()



    def forward(self, cat_arr, dt_arr, amount_arr, id_arr):
        x_history = self.transformer_encoder(cat_arr, dt_arr, amount_arr, id_arr)
        x_history = self.dropout(x_history)
        z_history = self.encoder_history(x_history).squeeze(2)

        return z_history

    def step(self, batch: Any, stage: str):
        batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr = batch
        if stage == 'train':
            preds = self.forward(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
            loss = multilabel_crossentropy_loss(preds, batch_current_cat, self.cat_vocab_size)
            return loss, preds

        preds = self.forward(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
        loss = multilabel_crossentropy_loss(preds, batch_current_cat, self.cat_vocab_size)
        batch_mask_current_cat = torch.tensor(~(batch_current_cat == self.cat_vocab_size), dtype=torch.int64).unsqueeze(2)
        batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                     num_classes=self.cat_vocab_size+1) * batch_mask_current_cat, dim=1)
        targets = batch_onehot_current_cat[:, :-1].detach().cpu().tolist()
        return loss, preds, targets


    def training_step(self, batch: Any, batch_idx: int):
        loss, preds  = self.step(batch, 'train')
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds}

    def training_epoch_end(self, outputs: List[Any]):
        pass
        # # `outputs` is a list of dicts returned from `training_step()`
        # all_preds = outputs[0]["preds"]
        # all_targets = outputs[0]["targets"]
        #
        # for i in range(1, len(outputs)):
        #     all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
        #     all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
        #
        #
        # self.log(
        #     "train/" + self.metric_name,
        #     self.criterion(all_targets, all_preds),
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, 'valid')
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        #print(outputs, len(outputs))
        all_preds = outputs[0]["preds"]
        all_targets = outputs[0]["targets"]

        self.final_thr, self.k = valid_thr(np.array(all_targets), np.array(all_preds))
        print('self.final_thr, self.k', self.final_thr, self.k)


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, 'test')
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        all_preds = outputs[0]["preds"]
        all_targets = outputs[0]["targets"]
        #all_rollings = outputs[0]["rolling"]

        # for i in range(1, len(outputs)):
        #     all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
        #     all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
        #     all_rollings = torch.cat((all_rollings, outputs[i]["rolling"]), 0)
        metrics_dict = calculate_all_metrics(np.array(all_targets), np.array(all_preds), self.final_thr, self.k, kind='thr')
        self.log(
            "test/" + 'f1_micro',
            metrics_dict['f1_micro'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/" + 'f1_macro',
            metrics_dict['f1_macro'],
            on_epoch=True,
            prog_bar=True,
        )


    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
