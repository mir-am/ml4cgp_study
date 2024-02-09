from typing import Union, Literal
from transformers import T5EncoderModel, AdamW, get_linear_schedule_with_warmup, AutoModel
from torch import nn
from torch.nn import functional as F
from src.constants import STRUCT_FEAT_DIM
from torchmetrics.classification import BinaryConfusionMatrix
import pytorch_lightning as pl
import torch
import warnings
warnings.filterwarnings('ignore')
# Faster model training
torch.set_float32_matmul_precision('medium')


class CodeT5CGPruner(pl.LightningModule):
    def __init__(self, num_train_samples, model_name: str, num_train_epochs=1,
                 mode: Union[Literal['balanced'], Literal['pruner'], Literal['adder']] ="now",
                 lr=2e-4, dropout_rate=0.25, warmup_steps=100):
        super().__init__()
        self.encoder = self.get_clm_model(model_name)
        
        self.out = nn.Linear(self.encoder.config.hidden_size, 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.save_hyperparameters()

        if self.hparams.mode == ModelsMode.CUSTOM:
            self.ce_weight = None # Should be changed at run-time
        elif self.hparams.mode == ModelsMode.NOW: # No weights
            self.ce_weight = None
        else:
            raise RuntimeError(f"Model's weight, {self.hparams.mode}, is NOT recognized!")

        # Metrics
        self.bcm = BinaryConfusionMatrix(threshold=0.5)
        self.bcm_final = None

    def get_clm_model(self, model_name):
        if model_name == "codet5":
            return T5EncoderModel.from_pretrained('Salesforce/codet5-base')
        elif model_name == "codet5_plus":
            return T5EncoderModel.from_pretrained('Salesforce/codet5p-770m') #, torch_dtype=torch.bfloat16)
        elif model_name == "codebert":
            return AutoModel.from_pretrained("microsoft/codebert-base")
        else:
            raise RuntimeError("Model name not recognized! Choose either 'codebert' or 'codet5'")

    def forward(self,ids, mask):
        emb = self.encoder(ids, attention_mask=mask, return_dict=False)[0]
        emb = emb[:, -1]
        self.dropout(emb)
        out = self.out(emb)
        return out

    def common_step(self, batch, batch_idx, compute_bcm=True):
        outputs = self(batch['ids'], batch['mask'])
        labels = batch['label']
        loss = F.cross_entropy(outputs, batch['label'], weight=self.ce_weight)
        if compute_bcm:
            outputs = F.softmax(outputs)
            outputs = torch.argmax(outputs, dim=1)
            self.bcm(outputs, labels)
            p, r, f1 = self.obtain_eval_metrics()
            
            return loss, p, r, f1
        else:
            return loss

    def training_step(self, batch, batch_idx):
        loss, p, r, f1 = self.common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "Precision": p, "Recall": r, "F1": f1},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, compute_bcm=False)
        self.log_dict({"val_loss": loss}, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, p, r, f1 = self.common_step(batch, batch_idx)
        self.log_dict({"Precision": p, "Recall": r, "F1": f1},
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self(batch['ids'], batch['mask'])
        outputs = F.softmax(outputs)
        outputs = outputs.detach().cpu().numpy()[:, 1] # probabilities
        return outputs

    def get_test_conf_mat(self):
        return self.bcm_final
    
    def obtain_eval_metrics(self):
        # Precision and recall for the CG prunning task
        cm = self.bcm.compute()
        r = cm[1,1] / (cm[1,1]+cm[1,0])
        p = cm[1,1] / (cm[1,1]+cm[0,1])
        f1 = 2*p*r/(p+r)
        return p, r, f1

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        num_train_optimization_steps = self.hparams.num_train_epochs * self.hparams.num_train_samples
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def set_ce_weight(self, ce_weight: torch.cuda.FloatTensor):
        self.ce_weight = ce_weight


class CLMWithStructFeat(CodeT5CGPruner):
    def __init__(self, num_train_samples, model_name: str, num_train_epochs=1,
                 mode: Union[Literal['balanced'], Literal['pruner'], Literal['adder']] ="balanced",
                 lr=2e-4, dropout_rate=0.25, warmup_steps=100, hidden_size=16):
        super().__init__(num_train_samples=num_train_samples, model_name=model_name,
                         num_train_epochs=num_train_epochs, mode=mode, lr=lr, dropout_rate=dropout_rate,
                         warmup_steps=warmup_steps)
        
        self.encoder = self.get_clm_model(model_name)
        
        self.code_feat_nn = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.struct_feat_nn = nn.Linear(STRUCT_FEAT_DIM, hidden_size)
        self.decoder = nn.Linear(2 * hidden_size, 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.save_hyperparameters()

        if self.hparams.mode == ModelsMode.CUSTOM:
            self.ce_weight = None # Should be changed at run-time
        elif self.hparams.mode == ModelsMode.NOW: # No weights
            self.ce_weight = None
        else:
            raise RuntimeError(f"Model's weight, {self.hparams.mode}, is NOT recognized!")

        self.bcm = BinaryConfusionMatrix(threshold=0.5)
        self.bcm_final = None

    def get_clm_model(self, model_name):
        if model_name == "codet5_ws":
            return T5EncoderModel.from_pretrained('Salesforce/codet5-base')
        elif model_name == "codet5_plus_ws":
            return T5EncoderModel.from_pretrained('Salesforce/codet5p-770m') #, torch_dtype=torch.bfloat16)
        elif model_name == "codebert_ws":
            return AutoModel.from_pretrained("microsoft/codebert-base")
        else:
            raise RuntimeError("Model name not recognized! Choose either 'codebert' or 'codet5'")

    def forward(self, ids, struct_feat, mask):
        emb = self.encoder(ids, attention_mask=mask, return_dict=False)[0]
        emb = emb[:, -1]
        h_c = self.code_feat_nn(emb)
        h_s = self.struct_feat_nn(struct_feat)
        h = torch.cat([h_c, h_s], axis=1)
        # h = self.dropout(h)
        out = self.decoder(h)
    
        return out
    
    def common_step(self, batch, batch_idx, compute_bcm=True):
        outputs = self(batch['ids'], batch['struct_feat'], batch['mask'])
        labels = batch['label']
        loss = F.cross_entropy(outputs, batch['label'], weight=self.ce_weight)
        if compute_bcm:
            outputs = F.softmax(outputs)
            outputs = torch.argmax(outputs, dim=1)
            self.bcm(outputs, labels)
            p, r, f1 = self.obtain_eval_metrics()
            return loss, p, r, f1
        else:
            return loss
        
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self(batch['ids'], batch['struct_feat'], batch['mask'])
        outputs = F.softmax(outputs)
        outputs = outputs.detach().cpu().numpy()[:, 1] # probabilities
        return outputs

class ModelsMode:
    """
    Weights for training the models, i.e., cross entropy loss function
    """
    NOW = "now" # No weights
    CUSTOM = 'custom' # Name can be changed at run time
