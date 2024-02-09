from typing import Tuple
from torch.utils.data import DataLoader
from src.model import CodeT5CGPruner, CLMWithStructFeat
from tqdm import tqdm
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
import numpy as np

def predict_dataset(model_chkp_path: str, predict_dataloader: DataLoader, model_name="") -> Tuple[np.array, np.array]:

    t = pl.Trainer(accelerator='gpu', precision=16)
    if "ws" not in model_name:
        model = CodeT5CGPruner.load_from_checkpoint(model_chkp_path)
    else:
        model = CLMWithStructFeat.load_from_checkpoint(model_chkp_path)
    model = torch.compile(model)
    preds = t.predict(model, dataloaders=predict_dataloader)

    preds_prob = np.hstack(preds)
    preds_labels = np.where(preds_prob >= 0.5, 1, 0)

    return preds_prob, preds_labels

def predict_dataset_w_model(model: CodeT5CGPruner, trainer: pl.Trainer, predict_dataloader: DataLoader) -> Tuple[np.array, np.array]:
    preds = trainer.predict(model, dataloaders=predict_dataloader)

    preds_prob = np.hstack(preds)
    preds_labels = np.where(preds_prob >= 0.5, 1, 0)

    return preds_prob, preds_labels

def predict_w_pt(model, test_dataloader: DataLoader, limit_batches:int=-1):
    model.to("cuda")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), leave=False, total=len(test_dataloader)):
            ids = batch['ids'].to("cuda")
            mask = batch['mask'].to("cuda")
            output = model(ids, mask)
            output = F.softmax(output)
            preds = output.detach().cpu().numpy()[:, 1]
            #preds = np.where(output >= 0.5, 1, 0)
            labels = batch['label'].detach().cpu().numpy()
            all_preds = np.concatenate((all_preds, preds))
            all_labels = np.concatenate((all_labels, labels))
            if i == limit_batches:
                break

def predict_one_sample(model, one_batch: dict) -> np.array:
    model.to("cuda")
    model.eval()

    ids = one_batch['ids'].unsqueeze(0).to("cuda")
    mask = one_batch['mask'].unsqueeze(0).to("cuda")
  
    output = model(ids, mask)
    output = F.softmax(output)
    preds = output.detach().cpu().numpy()[:, 1]
    return preds