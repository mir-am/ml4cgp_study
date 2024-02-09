"""
Evaluates trained models on a dataset
"""

from torch.utils.data import DataLoader
from src.dataset import DatasetTypes, load_dataset_files, create_dataset
from src import eval
from src.test import predict_dataset
from os.path import join
import pandas as pd
import torch

TEST_PROGRAMS_LIST = "/mnt/data/amir_projects/ml4cg/data/njr1/test_programs.txt"
PROJECT_FOLDER = "/mnt/data/amir_projects/ml4cg/"
FP_PRECISON = 16

datasets= [DatasetTypes.ORG_FIXED]
models = ['codet5', 'codet5_plus']

models_chkp_path = {'codebert': "/mnt/data/amir_projects/ml4cg/model/codebert_pruner_org/lightning_logs/version_0/checkpoints/epoch=1-step=53744.ckpt",
                    'codet5': "/mnt/data/amir_projects/ml4cg/model/codet5_pruner_org/lightning_logs/version_1/checkpoints/epoch=1-step=53744.ckpt",
                    'codet5_plus': "/mnt/data/amir_projects/ml4cg/model/codet5_plus_pruner_org/lightning_logs/version_0/checkpoints/epoch=1-step=143316.ckpt"}

with open(TEST_PROGRAMS_LIST, 'r') as f:
    test_programs_list = f.read().splitlines()

for d in datasets:
    for m in models:
        print("*"*20 + f"Training and evaluating {m} on {d.value}" + "*"*20)
        _, _, test_path = load_dataset_files(d)
        df_test = pd.read_csv(join(PROJECT_FOLDER, test_path))
        test_dataset_df = df_test
        test_dataset_name = f"test_{d.value}"
        test_dataset = create_dataset(df_test, join(PROJECT_FOLDER, "./data/njr1/"), test_dataset_name, 0)
        test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=256)
        print(f"No. of test batches {len(test_dataloader):,}")

        # model_trained = CodeT5CGPruner.load_from_checkpoint(models_chkp_path[m])
        # m_trained_compiled = torch.compile(model_trained)

        df_test['m_out_p'], df_test['m_out'] = predict_dataset(models_chkp_path[m], test_dataloader)

        test_programs_results_df = eval.compute_eval_metrics(test_programs_list, df_test)
        eval.report_eval_metrics(test_programs_results_df)
