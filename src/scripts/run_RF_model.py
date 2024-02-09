from src.dataset import DatasetTypes, load_dataset_files
from src.dataset import get_X_Y_arrays
from src.eval import compute_eval_metrics, report_eval_metrics, compute_eval_metrics_paper, report_eval_metrics_paper
from src.model import ModelsMode
from src import PROJECT_DATA_PATH
from os.path import join, exists
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import math
import os
import pickle
import numpy as np
import pandas as pd
import psutil
import time
import sklearn
import faulthandler
faulthandler.enable()

print(f"Sklearn verison: {sklearn.__version__}")

CUT_OFF_VALUE = 0.45 # Used in CGPruner & AutoPruner
NO_CPU_CORES = math.floor(0.9 * psutil.cpu_count(logical=False)) # For training
FORCE_TRAINING = False
NJR1_RAW_DATA = join(PROJECT_DATA_PATH, "njr1/raw_data")
XCORPUS_DATASET = join(PROJECT_DATA_PATH, "xcorpus/dataset/")
YCORPUS_DATASET = join(PROJECT_DATA_PATH, "ycorpus/dataset/")
RESULTS_FOLDER = join(PROJECT_DATA_PATH, "results_new")
RF_MODELS = join(PROJECT_DATA_PATH, "models/RF")
os.makedirs(RF_MODELS, exist_ok=True)

def load_dataset(dataset_name):
    train_path, valid_path, test_path, _ = load_dataset_files(dataset_name)
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)
    df_train_valid = pd.concat([df_train, df_valid])
    return df_train_valid, df_test

def make_nyx_corpus():
    dataset_name = DatasetTypes.ORG_W_STRUCT
    df_train_njr1, df_test_njr1 = load_dataset(dataset_name)
    X_train_njr1, Y_train_njr1, _, _, _, _ = get_X_Y_arrays(NJR1_RAW_DATA, df_train_njr1)
    X_test_njr1, Y_test_njr1, Y_test_wala_njr1, test_programs_list_njr1, _, njr1_test_df = get_X_Y_arrays(NJR1_RAW_DATA, df_test_njr1)

    dataset_name = DatasetTypes.XCORP_W_STRUCT
    df_train_xc, df_test_xc = load_dataset(dataset_name)
    X_train_xc, Y_train_xc, _, _, _, _ = get_X_Y_arrays(XCORPUS_DATASET, df_train_xc, "v4-2")
    X_test_xc, Y_test_xc, Y_test_wala_xc, test_programs_list_xc, _, xc_test_df = get_X_Y_arrays(XCORPUS_DATASET, df_test_xc, "v4-2")

    dataset_name = DatasetTypes.DHTC_W_STRUCT
    df_train_yc, df_test_yc = load_dataset(dataset_name)
    X_train_yc, Y_train_yc, _, _, _, _ = get_X_Y_arrays(YCORPUS_DATASET, df_train_yc, "v7-3-4")
    X_test_yc, Y_test_yc, Y_test_wala_yc, test_programs_list_yc, _, yc_test_df = get_X_Y_arrays(YCORPUS_DATASET, df_test_yc, "v7-3-4")

    nyx_test_df = pd.concat([njr1_test_df, xc_test_df, yc_test_df], ignore_index=True)

    return np.vstack((X_train_njr1, X_train_xc, X_train_yc)), np.hstack((Y_train_njr1, Y_train_xc, Y_train_yc)),\
           np.vstack((X_test_njr1, X_test_xc, X_test_yc)), np.hstack((Y_test_njr1, Y_test_xc, Y_test_yc)),\
           np.hstack((Y_test_wala_njr1, Y_test_wala_xc, Y_test_wala_yc)), \
           test_programs_list_njr1 + test_programs_list_xc + test_programs_list_yc, nyx_test_df

def main():

    for dataset_name in [DatasetTypes.ORG_W_STRUCT, DatasetTypes.XCORP_W_STRUCT, DatasetTypes.DHTC_W_STRUCT, DatasetTypes.NYX_W_STRUCT]:
        model_mode = ModelsMode.NOW

        if 'nyx' in dataset_name.value:
            X_train, Y_train, X_test, Y_test, Y_test_wala, test_programs_list, df_test = make_nyx_corpus()
        else:
            train_path, valid_path, test_path, _ = load_dataset_files(dataset_name)
            df_train = pd.read_csv(train_path)
            df_valid = pd.read_csv(valid_path)
            df_test = pd.read_csv(test_path)
            df_train_valid = pd.concat([df_train, df_valid])

            if 'xcorpus' in dataset_name.value:
                X_train, Y_train, Y_train_wala, train_programs_list, m2t_l, _ = get_X_Y_arrays(XCORPUS_DATASET, df_train_valid, "v4-2")
                X_test, Y_test, Y_test_wala, test_programs_list, m2t_l, df_test = get_X_Y_arrays(XCORPUS_DATASET, df_test, "v4-2")
            elif 'dhtc' in dataset_name.value:
                X_train, Y_train, Y_train_wala, train_programs_list, m2t_all, _ = get_X_Y_arrays(YCORPUS_DATASET, df_train_valid, "v7-3-4")
                X_test, Y_test, Y_test_wala, test_programs_list, m2t_all, df_test = get_X_Y_arrays(YCORPUS_DATASET, df_test, "v7-3-4")
            else: # NJR-1
                X_train, Y_train, Y_train_wala, train_programs_list, m2t_all, _ = get_X_Y_arrays(NJR1_RAW_DATA, df_train_valid)
                X_test, Y_test, Y_test_wala, test_programs_list, m2t_all, df_test = get_X_Y_arrays(NJR1_RAW_DATA, df_test)

        print(f"No. of training and test samples, {X_train.shape[0]:,}, {X_test.shape[0]:,}")

        # Normalization
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Training
        param_grid = {
            'n_estimators': [100, 250, 500, 1000],
            'max_depth': [None, 10, 25, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        model_path = join(RF_MODELS, f'cg_pruner_rf_best_{model_mode}_{dataset_name.value}.pkl')
        model_pruner_path = join(RF_MODELS, f'cg_pruner_rf_best_pruner_{dataset_name.value}.pkl') # Legacy name
        if exists(model_path):
            pass
        elif exists(model_pruner_path):
            model_path = model_pruner_path
    
        if not exists(model_path) or FORCE_TRAINING:
            print("*"*20 + f"Training and evaluating RF on mode {model_mode} {dataset_name.value}" + "*"*20)

            clf = RandomForestClassifier(
                max_features = "sqrt",
                class_weight=None,
                random_state = 0,
                bootstrap = False,
                criterion = "entropy",
                n_jobs=NO_CPU_CORES,
                #verbose=1
                )

            grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=KFold(n_splits=4), verbose=2)
            start_time = time.time()
            grid_search.fit(X_train, Y_train)
            end_time = time.time()
            print(f'Best parameters: {grid_search.best_params_}')
            print(f'Time elapsed: {end_time - start_time} seconds')

            # Saving the model
            with open(model_path, 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)

            y_pred_proba = grid_search.predict_proba(X_test)
            y_pred_proba = y_pred_proba[:,0]
            y_preds = np.where(y_pred_proba > 0.45, 0, 1)

            # Evaluation
            df_test = df_test[df_test['method'] != "<boot>"]
            df_test_results = pd.DataFrame({'wiretap': Y_test, 'wala-cge-0cfa-noreflect-intf-trans': Y_test_wala,
                                            'method': df_test['method'], 'target': df_test['target'],
                                            'm_out_p': y_pred_proba, 'm_out': y_preds, 'program_name': test_programs_list})

            print(f"Evaluation results for RF with mode {model_mode} on dataset {dataset_name.value}")
            test_programs_results_df = compute_eval_metrics(test_programs_list, df_test_results, model_out_col='m_out')
            report_eval_metrics(test_programs_results_df)

            df_test_results.to_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_RF_CG_pruner_best_{model_mode}_{dataset_name.value}.csv"))
            test_programs_results_df.to_csv(join(RESULTS_FOLDER, f"test_programs_metrics_RF_CG_pruner_best_{model_mode}_{dataset_name.value}.csv"))
        else:
            print("*"*20 + f"Only evaluating RF on mode {model_mode} {dataset_name.value}" + "*"*20)
            with open(model_path, 'rb') as f:  
                model = pickle.load(f)
                y_pred_proba = model.predict_proba(X_test)
                y_pred_proba = y_pred_proba[:,0]
                y_preds = np.where(y_pred_proba > 0.45, 0, 1)
            
            # Evaluation
            df_test = df_test[df_test['method'] != "<boot>"]
            df_test_results = pd.DataFrame({'wiretap': Y_test, 'wala-cge-0cfa-noreflect-intf-trans': Y_test_wala,
                                            'method': df_test['method'], 'target': df_test['target'],
                                            'm_out_p': y_pred_proba, 'm_out': y_preds, 'program_name': test_programs_list})

            print(f"Evaluation results for RF with mode {model_mode} on dataset {dataset_name.value}")
            test_programs_results_df = compute_eval_metrics_paper(test_programs_list, df_test_results, model_out_col='m_out')
            report_eval_metrics_paper(test_programs_results_df)

            df_test_results.to_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_RF_CG_pruner_best_{model_mode}_{dataset_name.value}.csv"))
            test_programs_results_df.to_csv(join(RESULTS_FOLDER, f"test_programs_metrics_RF_CG_pruner_best_{model_mode}_{dataset_name.value}.csv"))

if __name__ == "__main__":
    main()