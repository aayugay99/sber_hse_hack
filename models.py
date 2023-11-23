from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import AUROC


class RNNModel(pl.LightningModule):
    def __init__(self, seq_encoder, optimizer_partial, lr_scheduler_partial, head_hidden=512, dropout=0.1):
        super().__init__()


        self.seq_encoder = seq_encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(seq_encoder.embedding_size),
            nn.Linear(seq_encoder.embedding_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(head_hidden),
            nn.Linear(head_hidden, 1)
        ) 

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        self.metric = {"train": AUROC(task="binary"), "valid": AUROC(task="binary")} 
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X):
        embeddings = self.seq_encoder(X)

        if not self.seq_encoder.is_reduce_sequence:
            # mean pool
            embeddings = embeddings.payload.sum(dim=1)
            embeddings /= X.seq_lens.unsqueeze(1).expand_as(embeddings)

        logits = self.head(embeddings).squeeze()
        return logits

    def shared_step(self, stage, batch, _):
        X, y = batch

        logits = self(X)

        loss = None
        if stage == 'train':
            loss = self.loss(logits, y.float())

        self.metric[stage].update(logits, y.long())
        self.log(f'{stage}_auc', self.metric[stage].compute(), prog_bar=True)

        return loss

    def training_step(self, *args, **kwargs):
        return self.shared_step('train', *args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return self.shared_step('valid', *args, **kwargs)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        logits = self(batch)
        return nn.functional.sigmoid(logits)

    @property
    def metric_name(self):
        return 'valid_auc'

    def on_train_epoch_end(self):
        self.metric["train"].reset()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        self.log('valid_auc', self.metric["valid"].compute(), prog_bar=True)
        self.metric["valid"].reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]


class BOTModel(pl.LightningModule):
    def __init__(self, trx_encoder, optimizer_partial, lr_scheduler_partial, head_hidden=512, dropout=0.1):
        super().__init__()

        self.trx_encoder = trx_encoder

        self.query = nn.Parameter(torch.randn(trx_encoder.output_size), requires_grad=True)
        self.attn = nn.MultiheadAttention(trx_encoder.output_size, 4, batch_first=True)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(trx_encoder.output_size),
            nn.Linear(trx_encoder.output_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(head_hidden),
            nn.Linear(head_hidden, 1)
        ) 

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        self.metric = {"train": AUROC(task="binary"), "valid": AUROC(task="binary")} 
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X):
        embeddings = self.trx_encoder(X).payload

        attn_output, _ = self.attn(
            self.query.unsqueeze(0).expand(embeddings.shape[0], 1, -1), 
            embeddings,
            embeddings,
            key_padding_mask=(1-X.seq_len_mask).bool()
        )

        logits = self.head(attn_output.squeeze()).squeeze()

        return logits

    def shared_step(self, stage, batch, _):
        X, y = batch

        logits = self(X)

        loss = None
        if stage == 'train':
            loss = self.loss(logits, y.float())

        self.metric[stage].update(logits, y.long())
        self.log(f'{stage}_auc', self.metric[stage].compute(), prog_bar=True)

        return loss

    def training_step(self, *args, **kwargs):
        return self.shared_step('train', *args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return self.shared_step('valid', *args, **kwargs)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        logits = self(batch)
        return nn.functional.sigmoid(logits)

    @property
    def metric_name(self):
        return 'valid_auc'

    def on_train_epoch_end(self):
        self.metric["train"].reset()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        self.log('valid_auc', self.metric["valid"].compute(), prog_bar=True)
        self.metric["valid"].reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]