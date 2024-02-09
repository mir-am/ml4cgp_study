"""
Trains and evaluates a selected model on a set of datasets
"""
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from src.dataset import DatasetTypes, load_dataset_files, create_dataset, ignore_boot_methods, ignore_boot_methods_df
from src.utils import find_model_checkpoint
from src.test import predict_dataset
from src import PROJECT_DATA_PATH, TESTING_MODE
from os.path import join
import pandas as pd
import numpy as np

import argparse
import faulthandler

from src import model
from src.model import ModelsMode
from src import eval
import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')
faulthandler.enable()

print(f"Torch version: {torch.__version__} Lighting v.{pl.__version__}")

TRAIN_PROGRAMS_LIST = join(PROJECT_DATA_PATH, "njr1/train_programs.txt")
VALID_PROGRAMS_LIST = join(PROJECT_DATA_PATH, "njr1/valid_programs.txt")
TEST_PROGRAMS_LIST = join(PROJECT_DATA_PATH, "njr1/test_programs.txt")
MODEL_CHK_POINTS = join(PROJECT_DATA_PATH, "models")
RESULTS_FOLDER = join(PROJECT_DATA_PATH, "results_new")
FORCE_TRAINING = False

MODELS_BATCH_SIZE = {"codebert": 32, "codebert_ws": 32, "codet5": 32, "codet5_ws": 32, "codet5_plus": 12, "codet5_plus_ws": 12}
NO_EPOCHS = 2
FP_PRECISON = 16
VAL_CHK_INTERVAL = None
LR = 0.00001 # Learning rate
TESTING_MODE_SAMPLES = 1000


def get_dataset_subset(full_dataset: Dataset, full_dataset_df: pd.DataFrame):
    """
    Only used for testing the code
    """
    # 10% of data
    # int(0.05 * len(full_dataset))
    subset_size = 1000
    indices = torch.randperm(len(full_dataset)).tolist()
    subset_indices = indices[:subset_size]
    dataset_subset = Subset(full_dataset, subset_indices)
    dataset_subset_df = full_dataset_df.iloc[subset_indices]
    return dataset_subset, dataset_subset_df

def train_and_evaluate_model(dataset, model_name, model_mode, ce_weight: torch.cuda.FloatTensor =None):
    model_full_name = model_name + ("_ws" if "w_struct" in dataset.value else "")
    train_path, valid_path, test_path, dataset_path = load_dataset_files(dataset)
    df_test = pd.read_csv(test_path)
    df_test = ignore_boot_methods_df(df_test)
    df_test.reset_index(inplace=True)
    if TESTING_MODE:
        df_test = df_test.sample(TESTING_MODE_SAMPLES)
        df_test.reset_index(drop=True, inplace=True)
        print(f"Testing mode with {TESTING_MODE_SAMPLES} samples")

    test_dataset_df = df_test
    test_dataset_name = f"test_{dataset.value}"
    test_dataset = create_dataset(test_dataset_df, dataset_path,
                            test_dataset_name, model_name, 2, False)

    print(f"No. of test samples/edges: {len(test_dataset):,}")

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=64)
    print(f"No. of test batches {len(test_dataloader):,}")

    model_chk_path = join(MODEL_CHK_POINTS, f'{model_full_name}_pruner_{model_mode}_{dataset.value}')
    model_mode = model_mode #'now'

    m_chk_point = find_model_checkpoint(model_chk_path, f"epoch={NO_EPOCHS-1}-step=*.ckpt")
    if m_chk_point is None and model_mode == ModelsMode.NOW:
        m_chk_point = find_model_checkpoint(model_chk_path.replace("_now", ""), f"epoch={NO_EPOCHS-1}-step=*.ckpt")
    if m_chk_point is None:
        m_chk_point = find_model_checkpoint(model_chk_path, f"epoch=0-step=*.ckpt")

    trainer = None
    if m_chk_point is None or FORCE_TRAINING:
        # Training
        print("*"*20 + f"Training and evaluating {model_name} on mode {model_mode} {dataset.value}" + "*"*20)
        df_train = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)
        df_train, df_valid = ignore_boot_methods_df(df_train), ignore_boot_methods_df(df_valid)
        train_dataset_df = df_train
        train_dataset_name = f"train_{dataset.value}"
        valid_dataset_df = df_valid
        valid_dataset_name = f"valid_{dataset.value}"

        train_dataset = create_dataset(train_dataset_df, dataset_path,
                                train_dataset_name, model_name, 0)
        valid_dataset = create_dataset(valid_dataset_df, dataset_path,
                                valid_dataset_name, model_name, 1)
        
        print(f"No. of traning samples/edges: {len(train_dataset):,}")
        print(f"No. of validation samples/edges: {len(valid_dataset):,}")
        train_valid_dataset = ConcatDataset([train_dataset, valid_dataset])
        train_dataloader = DataLoader(train_valid_dataset, num_workers=4,
                        batch_size=MODELS_BATCH_SIZE[model_full_name], pin_memory=True, shuffle=True)
        print(f"No. of training batches {len(train_dataloader):,}")

        if "ws" not in model_full_name:
            m = model.CodeT5CGPruner(num_train_samples=len(train_dataloader),
                            model_name=model_full_name,
                            num_train_epochs=NO_EPOCHS,
                            mode=model_mode,
                            lr=LR)
        else:
            m = model.CLMWithStructFeat(num_train_samples=len(train_dataloader),
                            model_name=model_full_name,
                            num_train_epochs=NO_EPOCHS,
                            mode=model_mode,
                            lr=LR)
        if model_mode == ModelsMode.CUSTOM and ce_weight is not None:
            print("Using a custom weight")
            m.set_ce_weight(ce_weight)
        m_compiled = torch.compile(m)
        trainer = pl.Trainer(accelerator="gpu", max_epochs=NO_EPOCHS, default_root_dir=model_chk_path,
                    fast_dev_run=False, precision=FP_PRECISON, #logger=wandb_logger_p,
                    limit_train_batches=1.0) #s, strategy="deepspeed_stage_2_offload") #, limit_val_batches=0.05)
        trainer.fit(model=m_compiled, train_dataloaders=train_dataloader) #, val_dataloaders=valid_dataloader)
        # Evaluation
        preds = trainer.predict(dataloaders=test_dataloader)

        df_test['m_out_p'] = np.hstack(preds)
        df_test['m_out'] = np.where(df_test['m_out_p'].to_numpy() >= 0.5, 1, 0)
    else:
        print(f"Found a model checkpoint at {m_chk_point}")
        print("*"*20 + f"Evaluating {model_name} on mode {model_mode} {dataset.value}" + "*"*20)
        preds_prob, preds_labels = predict_dataset(m_chk_point, test_dataloader, model_full_name)
        df_test['m_out_p'] = preds_prob
        df_test['m_out'] = preds_labels

    test_programs_results_df = eval.compute_eval_metrics_paper([], df_test)
    eval.report_eval_metrics_paper(test_programs_results_df)

    df_test.to_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{model_full_name}_pruner_{model_mode}_{dataset.value}.csv"), index=False)
    test_programs_results_df.to_csv(join(RESULTS_FOLDER, f"test_programs_metrics_{model_full_name}_pruner_{model_mode}_{dataset.value}.csv"), index=False)

def train_and_evaluate(models, datasets, modes, force_training=False):
    for d in datasets:
        for mode in modes:
            for m_name in models:
                train_and_evaluate_model(d, m_name, mode)
         
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training & evaluating models")
    parser.add_argument('rq', choices=['RQ1', 'RQ2', 'RQ3'], help='Research Questions')
    args = parser.parse_args()

    if args.rq == "RQ1":
        # RQ1
        train_and_evaluate(['codebert'], [DatasetTypes.ORG, DatasetTypes.XCORP, DatasetTypes.DHTC_W_SRC,
                                          DatasetTypes.NYX_W_SRC], [ModelsMode.NOW], force_training=False)
        # AutoPruner
        train_and_evaluate(['codebert'], [DatasetTypes.XCORP_W_STRUCT, DatasetTypes.DHTC_W_STRUCT,
                                          DatasetTypes.NYX_W_STRUCT, DatasetTypes.ORG_W_STRUCT],
                                                [ModelsMode.NOW], force_training=False)
        train_and_evaluate(['codet5'], [DatasetTypes.ORG, DatasetTypes.XCORP, DatasetTypes.DHTC_W_SRC,
                                          DatasetTypes.NYX_W_SRC], [ModelsMode.NOW], force_training=False)
        train_and_evaluate(['codet5_plus'], [DatasetTypes.ORG, DatasetTypes.XCORP, DatasetTypes.DHTC_W_SRC,
                                        DatasetTypes.NYX_W_SRC], [ModelsMode.NOW], force_training=False)
        
    elif args.rq == "RQ2":
        # RQ2
        for w in [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
            model_mode_custom_name = f"C{str(w).replace('.', '')}"
            ModelsMode.CUSTOM = model_mode_custom_name
            print(f"Training/evaluating with mode {ModelsMode.CUSTOM}")
            train_and_evaluate_model(DatasetTypes.NYX_W_SRC, 'codebert', ModelsMode.CUSTOM,
                                      torch.cuda.FloatTensor([1.0-w, w]))
            train_and_evaluate_model(DatasetTypes.NYX_W_SRC, 'codet5', ModelsMode.CUSTOM,
                                      torch.cuda.FloatTensor([1.0-w, w]))
    
    elif args.rq == "RQ3":
        # RQ3
        train_and_evaluate(['codet5'], [DatasetTypes.NYX_ONE_CFA], [ModelsMode.NOW])
        train_and_evaluate(['codebert'], [DatasetTypes.NYX_ONE_CFA], [ModelsMode.NOW])
    else:
        raise RuntimeError("Incorrect RQ number given! Choose between [RQ1, RQ2, RQ3]")
