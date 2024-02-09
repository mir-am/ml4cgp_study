"""
Measuring the features extraction time and the models inference
"""

from src import dataset
from typing import List
from torch.utils.data import DataLoader
from src.dataset import load_dataset_files
from src.model import CodeT5CGPruner
from src import PROJECT_DATA_PATH, TESTING_MODE
from tqdm import tqdm
from os.path import join
import pandas as pd
import pytorch_lightning as pl
import torch
import time
import json
import statistics
import faulthandler
faulthandler.enable()

OUTPUT_FOLDER = join(PROJECT_DATA_PATH, "results_new")
TESTING_SAMPLES = 500

def gen_feat_bench(dataset_name, feature_name, output_file_name):
    train_path, valid_path, test_path, _ = load_dataset_files(dataset_name)
    df_train = pd.read_csv(join(PROJECT_DATA_PATH, train_path))
    df_valid = pd.read_csv(join(PROJECT_DATA_PATH, valid_path))
    df_test = pd.read_csv(join(PROJECT_DATA_PATH, test_path))
    df_test = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    if TESTING_MODE:
        df_test = df_test.sample(TESTING_SAMPLES)
        df_test.reset_index(drop=True, inplace=True)
        print(f"Testing mode with {TESTING_SAMPLES:,} samples")
    df_test_programs = df_test['program_name'].unique()
    df_test_programs_idx = [df_test['program_name'] == p for p in df_test_programs]

    test_progs_feat_ext_exec_t = []
    feat_ext_exec_progs = {}
    s_t_feat_ext = None
    e_t_feat_ext = None
    for i, p in tqdm(enumerate(df_test_programs), total=len(df_test_programs)):
        p_test_samples = df_test[df_test_programs_idx[i]].reset_index(drop=True)
        if feature_name == "f_sem":
            s_t_feat_ext = time.time()
            dataset.CallGraphDataset(p_test_samples, save_data_path="", mode="test",
                                            on_fly_process=True, disable_pb=True)
            e_t_feat_ext = time.time() - s_t_feat_ext
   
        test_progs_feat_ext_exec_t.append(e_t_feat_ext)
        feat_t_progs = [t for t in test_progs_feat_ext_exec_t]
        feat_ext_exec_progs[p] = e_t_feat_ext

    print(f"{feature_name} avg per prog {statistics.mean(feat_t_progs):.2f}+- {statistics.stdev(feat_t_progs):.2f} sec")
    with open(join(OUTPUT_FOLDER, output_file_name), "w") as f:
        json.dump(feat_ext_exec_progs, f, indent=4)
   
def run_model_inf(output_file_name: str, datasets: List[dataset.DatasetTypes]):
    # datasets= [dataset.DatasetTypes.NYX_W_SRC]
    models = ['codebert', 'codet5']
    models_chkp_path = {'codebert': join(PROJECT_DATA_PATH, "models/codebert_pruner_now_nyx_w_src/lightning_logs/version_0/checkpoints/epoch=1-step=80788.ckpt"),
                        'codet5': join(PROJECT_DATA_PATH, "models/codet5_pruner_now_nyx_w_src/lightning_logs/version_0/checkpoints/epoch=1-step=80788.ckpt")}
    models_inf_t = {'codebert': {}, 'codet5': {}}
    for d in datasets:
        train_path, valid_path, test_path, _ = load_dataset_files(d)
        df_train = pd.read_csv(join(PROJECT_DATA_PATH, train_path))
        df_valid = pd.read_csv(join(PROJECT_DATA_PATH, valid_path))
        df_test = pd.read_csv(join(PROJECT_DATA_PATH, test_path))
        df_test = pd.concat([df_train, df_valid, df_test], ignore_index=True)
        if TESTING_MODE:
            df_test = df_test.sample(TESTING_SAMPLES)
            df_test.reset_index(drop=True, inplace=True)
            print(f"Testing mode with {TESTING_SAMPLES:,} samples")
        df_test_programs = df_test['program_name'].unique()
        df_test_programs_idx = [df_test['program_name'] == p for p in df_test_programs]
        for m in models:
            t = pl.Trainer(accelerator='gpu', precision=16)
            model = CodeT5CGPruner.load_from_checkpoint(models_chkp_path[m])
            model = torch.compile(model)

            for i, p in tqdm(enumerate(df_test_programs), total=len(df_test_programs)):
                print(f"[{m}] Inference/Pruning for {p}")
                p_test_samples = df_test[df_test_programs_idx[i]].reset_index(drop=True)
                p_test_dataset = dataset.CallGraphDataset(p_test_samples, save_data_path="", mode="test",
                                                          on_fly_process=True, disable_pb=True)
                test_dataloader = DataLoader(p_test_dataset, num_workers=2, batch_size=128)
                s_t = time.time()
                preds = t.predict(model, dataloaders=test_dataloader)
                e_t = time.time() - s_t
                print(f"[{m}] Inference done in {e_t} in sec.")
                models_inf_t[m][p] = e_t

    with open(join(OUTPUT_FOLDER, output_file_name), "w") as f:
        json.dump(models_inf_t, f, indent=4)

def main():
    gen_feat_bench(dataset.DatasetTypes.NYX_W_SRC, "f_sem", "feat_ext_t_nyx_w_src_0cfa.json")
    gen_feat_bench(dataset.DatasetTypes.NYX_ONE_CFA, "f_sem", "feat_ext_t_nyx_w_src_1cfa.json")
    
    run_model_inf("models_inf_t_nyx_w_src_0cfa.json", [dataset.DatasetTypes.NYX_W_SRC])
    run_model_inf("models_inf_t_nyx_w_src_1cfa.json", [dataset.DatasetTypes.NYX_ONE_CFA])

if __name__ == "__main__":
    main()
    