from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
import copy


DEVICE = torch.device("cpu")


class DistillBert(pl.LightningModule):
    def __init__(self, n_layers=6, n_att_heads: int = 6, hid_dim: int = 512):
        super().__init__()
        hid_dim = (hid_dim // n_att_heads + (hid_dim % n_att_heads != 0)) * n_att_heads
        self.teacher_model = BertForSequenceClassification.from_pretrained("mfuntowicz/bert-base-cased-finetuned-sst2")
        # Never wanna update the teacher
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        student_config = copy.deepcopy(self.teacher_model.config)
        student_config.update(
            {"num_hidden_layers": n_layers, "num_attention_heads": n_att_heads, "hidden_size": hid_dim}
        )
        self.student_model = BertForSequenceClassification(student_config)

    def forward(self, input_ids, token_type, attention_mask) -> None:
        self.student_model.eval()
        student_out = self.student_model(input_ids, attention_mask, token_type)
        self.student_model.train()
        return student_out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        input_ids, token_type, attention_mask, labels = batch

        student_out = self.student_model(input_ids, attention_mask, token_type)
        teacher_out = self.teacher_model(input_ids, attention_mask, token_type)

        loss = F.mse_loss(student_out.logits, teacher_out.logits)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        self.teacher_model.eval()
        self.student_model.eval()

        input_ids, token_type, attention_mask, labels = batch

        teacher_out = self.teacher_model(input_ids, attention_mask, token_type)
        student_out = self.student_model(input_ids, attention_mask, token_type)

        loss = F.mse_loss(student_out.logits, teacher_out.logits)
        student_preds = torch.argmax(student_out.logits, dim=-1)
        teacher_preds = torch.argmax(teacher_out.logits, dim=-1)

        self.teacher_model.train()
        self.student_model.train()
        return {"val_loss": loss, "student_preds": student_preds, "teacher_preds": teacher_preds}

    def _calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor, pref: str) -> Dict[str, torch.Tensor]:
        preds = torch.argmax(logits, dim=-1)
        # accuracy = torchmetrics.functional.accuracy(preds, labels)
        return {f"{pref}preds": preds, f"{pref}target": labels}

    def validation_epoch_end(self, val_step_outputs) -> Dict[str, float]:
        loss = torch.hstack([step["val_loss"] for step in val_step_outputs]).mean()

        student_preds = torch.hstack([step["student_preds"] for step in val_step_outputs])
        teacher_preds = torch.hstack([step["teacher_preds"] for step in val_step_outputs])
        target = torch.hstack([step["target"] for step in val_step_outputs])

        student_acc = torchmetrics.functional.accuracy(student_preds, target)
        teacher_acc = torchmetrics.functional.accuracy(teacher_preds, target)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc_student", student_acc, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_acc_teacher", teacher_acc, on_epoch=True, prog_bar=True, logger=True)

        return {"student_acc": student_acc, "teacher_acc": teacher_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=2e-5, weight_decay=0.01)
        return optimizer


class Bert(pl.LightningModule):
    def __init__(self, n_layers=6, n_att_heads: int = 6, hid_dim: int = 512):
        super().__init__()
        hid_dim = (hid_dim // n_att_heads + (hid_dim % n_att_heads != 0)) * n_att_heads
        teacher_model = BertForSequenceClassification.from_pretrained("mfuntowicz/bert-base-cased-finetuned-sst2")

        config = copy.deepcopy(teacher_model.config)
        config.update(
            {"num_hidden_layers": n_layers, "num_attention_heads": n_att_heads, "hidden_size": hid_dim}
        )
        self.model = BertForSequenceClassification(config)

    def forward(self, input_ids, token_type, attention_mask) -> None:
        self.model.eval()
        student_out = self.model(input_ids, attention_mask, token_type)
        self.model.train()
        return student_out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        input_ids, token_type, attention_mask, labels = batch

        student_out = self.model(input_ids, attention_mask, token_type)
        loss = F.cross_entropy(student_out.logits, labels)

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        self.model.eval()
        input_ids, token_type, attention_mask, labels = batch

        student_out = self.model(input_ids, attention_mask, token_type)

        loss = F.cross_entropy(student_out.logits, labels)
        preds = torch.argmax(student_out.logits, dim=-1)

        self.model.train()
        return {"val_loss": loss, "preds": preds}

    def validation_epoch_end(self, val_step_outputs) -> Dict[str, float]:
        loss = torch.hstack([step["val_loss"] for step in val_step_outputs]).mean()

        preds = torch.hstack([step["preds"] for step in val_step_outputs])
        target = torch.hstack([step["target"] for step in val_step_outputs])

        acc = torchmetrics.functional.accuracy(preds, target)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

        return {"acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        return optimizer


class SST2DataModule(pl.LightningDataModule):
    def __init__(self, seq_len: int = None, train_batch_size: int = 8, valid_batch_size: int = 8):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("mfuntowicz/bert-base-cased-finetuned-sst2")
        self.seq_len = seq_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

    def prepare_data(self) -> None:
        self.init_dataset = load_dataset("glue", "sst2")

    def setup(self, stage: Optional[str] = None) -> None:
        train = self.init_dataset["train"]["sentence"]
        val = self.init_dataset["validation"]["sentence"]
        data = [train, val]
        data = [self.tokenizer(ds, padding=True, truncation=True) for ds in data]
        data = [[torch.Tensor(x).type(torch.IntTensor) for x in ds.values()] for ds in data]

        labels = [self.init_dataset["train"]["label"], self.init_dataset["validation"]["label"]]
        labels = [torch.Tensor(x).type(torch.IntTensor) for x in labels]

        datasets = [TensorDataset(*d, l) for d, l in zip(data, labels)]
        self.train_ds, self.val_ds = datasets

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.train_batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.valid_batch_size)


if __name__ == "__main__":
    pl.seed_everything(42)

    model = DistillBert()

    data = SST2DataModule(train_batch_size=64, valid_batch_size=64)

    wandb_logger = WandbLogger(project="hse_dl_distillation", log_model=False)

    trainer = pl.Trainer(val_check_interval=0.1,
                         max_epochs=2,
                         gpus=1 if torch.cuda.is_available() else 0,
                         logger=wandb_logger)
    trainer.fit(model, datamodule=data)

    print("Done!")
