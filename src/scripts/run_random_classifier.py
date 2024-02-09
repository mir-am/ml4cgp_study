"""
A random classifier which prunes call graph edges with equal chance, i.e., 0.50.
"""

from src.dataset import DatasetTypes, load_dataset_files
from src import eval
from src import PROJECT_DATA_PATH
from os.path import join
import pandas as pd
import random

RESULTS_FOLDER = join(PROJECT_DATA_PATH, "results_new")
MODEL_NAME = "RC"

if __name__ == "__main__":
    dataset_names = {'nyx_w_src': "NYXCorpus", "org": "NJR-1", "dhtc_w_src": "YCorpus", "xcorpus": "XCorpus"}
    for dataset in [DatasetTypes.NYX_W_SRC, DatasetTypes.ORG, DatasetTypes.DHTC_W_SRC, DatasetTypes.XCORP]:
        # dataset = DatasetTypes.NYX_W_SRC
        _, _, test_path, _ = load_dataset_files(dataset)
        df_test = pd.read_csv(test_path)
        df_test = df_test[df_test['method'] != "<boot>"]
        df_test['m_out'] = df_test.apply(lambda x: random.choice([0, 1]), axis=1)
        df_test['m_out_p'] = 0.50 

        print(f"Random Classifer on the dataset: {dataset_names[dataset.value]}")
        test_programs_results_df = eval.compute_eval_metrics_paper([], df_test)
        prec, recall, f1, f2, wala_prec, wala_rec, wala_f1, wala_f2 = eval.report_eval_metrics_paper(test_programs_results_df)
        print("******************************************************************************")

        df_test.to_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{MODEL_NAME}_pruner_now_{dataset.value}.csv"), index=False)
        test_programs_results_df.to_csv(join(RESULTS_FOLDER, f"test_programs_metrics_{MODEL_NAME}_pruner_now_{dataset.value}.csv"), index=False)
