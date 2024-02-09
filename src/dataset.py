# from src.utils import ParallelExecutor
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, AutoTokenizer, convert_slow_tokenizer, PreTrainedTokenizerFast
from dgl.data.utils import save_info, load_info
from tqdm import tqdm
from enum import Enum
from typing import Dict
from multiprocessing import Pool
# from joblib import delayed
from os.path import join, exists
from src.utils import get_input_and_mask
from src.constants import struct_feat_names
from src import PROJECT_DATA_PATH, TOKENIZER_BATCH_SIZE
import concurrent
import pandas as pd
import numpy as np
import torch

SA_LABEL = "wala-cge-0cfa-noreflect-intf-trans"
FEATURES_TO_REMOVE = [
    "wala-cge-0cfa-noreflect-intf-trans#edge_disjoint_paths_from_main",
    "wala-cge-0cfa-noreflect-intf-trans#node_disjoint_paths_from_main"
]
MAX_SEQ_LENGTH = 512
MAX_WORKERS = 16
TOKENIZER = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

class CallGraphDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, save_data_path, mode, model_name="codebert",
                 on_fly_process=False, disable_pb=False, progress_bar_pos=0):
        self.max_length = MAX_SEQ_LENGTH
        self.mode = mode
        self.save_data_path = join(save_data_path, f"processed_dataset_{mode}.pkl")
        self.data_df = data_df
        self.disable_pb = disable_pb
        self.progress_bar_pos = progress_bar_pos
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small', use_fast=True)

        if not on_fly_process:
            if self.has_cache():
                self.load()
                if not len(self.data) == len(data_df):
                    print("Inconsistent data on the disk. Should be re-processed")
                    self.process()
                    self.save()
            else:
                self.process()
                self.save()
        else:
            self.process()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ids = self.data[index]
        mask = self.mask[index]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.labels[index], dtype=torch.long),
            'static': torch.tensor(self.static_ids[index], dtype=torch.long),
            }

    
    def process_chunk(self, chunk):
        data, mask, labels, static_ids = [], [], [], []
        for i in chunk:
            if self.data_df['method'][i] == "<boot>":
                continue
            src, dst, lb, sanity_check = (self.data_df['method_src'][i], self.data_df['target_src'][i],
                                          self.data_df['wiretap'][i], self.data_df[SA_LABEL][i])
            if self.mode != "train" or sanity_check == 1:
                token_ids, mask_ids = get_input_and_mask(src, dst, self.max_length, self.tokenizer)
                data.append(token_ids)
                mask.append(mask_ids)
                labels.append(lb)
                static_ids.append(sanity_check)
        return data, mask, labels, static_ids
    
    def process(self):
        self.data = []
        self.mask = []
        self.static_ids = []
        self.labels = []

        indices = list(range(len(self.data_df['wiretap'])))
        chunk_size = TOKENIZER_BATCH_SIZE
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(self.process_chunk, chunks),
                                total=len(chunks),
                                position=self.progress_bar_pos,
                                disable=self.disable_pb))

        for result in results:
            self.data.extend(result[0])
            self.mask.extend(result[1])
            self.labels.extend(result[2])
            self.static_ids.extend(result[3])

    def save(self):
        save_info(self.save_data_path, {'label': self.labels,
                                        'data': self.data,
                                        'mask': self.mask,
                                        'static_ids': self.static_ids,
                                        })

    def load(self):
        print("Loading data ...")
        info_dict = load_info(self.save_data_path)
        self.labels = info_dict['label']
        self.data = info_dict['data']
        self.mask = info_dict['mask']
        self.static_ids = info_dict['static_ids']

    def has_cache(self):
        if exists(self.save_data_path):
            print("Data exists")
            return True
        return False

class CallGraphDatasetWithStructFeat(CallGraphDataset):
    """
    It has both code and structural features.
    """
    def __init__(self, data_df: pd.DataFrame, save_data_path, mode, on_fly_process=False, disable_pb=False):
        super().__init__(data_df=data_df, save_data_path=save_data_path, mode=mode,
                          on_fly_process=on_fly_process, disable_pb=disable_pb)

    def process(self):
        self.data = [] # Code features
        self.struct_feat = []
        self.mask = []
        self.static_ids = []
        self.labels = []
        self.struct_features_tmp = self.data_df[struct_feat_names].to_numpy()

        indices = list(range(len(self.data_df['wiretap'])))
        chunk_size = TOKENIZER_BATCH_SIZE 
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(self.process_chunk, chunks),
                                total=len(chunks),
                                position=self.progress_bar_pos,
                                disable=self.disable_pb))

        for result in results:
            self.data.extend(result[0])
            self.struct_feat.extend(result[1])
            self.mask.extend(result[2])
            self.labels.extend(result[3])
            self.static_ids.extend(result[4])

    def process_chunk(self, chunk):
        data, struct_feat, mask, labels, static_ids = [], [], [], [], []
        for i in chunk:
            if self.data_df['method'][i] == "<boot>":
                continue
            src, dst, lb, sanity_check = (self.data_df['method_src'][i], self.data_df['target_src'][i],
                                          self.data_df['wiretap'][i], self.data_df[SA_LABEL][i])
            if self.mode != "train" or sanity_check == 1:
                token_ids, mask_ids = get_input_and_mask(src, dst, self.max_length, self.tokenizer)
                data.append(token_ids)
                struct_feat.append(self.struct_features_tmp[i])
                mask.append(mask_ids)
                labels.append(lb)
                static_ids.append(sanity_check)
        return data, struct_feat, mask, labels, static_ids

    def __getitem__(self, index):
        ids = self.data[index]
        struct_feats = self.struct_feat[index]
        mask = self.mask[index]
        return {
            'ids': torch.tensor(ids, dtype=torch.long), # Code feature
            'struct_feat': torch.tensor(struct_feats, dtype=torch.float), # Structural feature
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.labels[index], dtype=torch.long),
            'static': torch.tensor(self.static_ids[index], dtype=torch.long),
            }

    def save(self):
        save_info(self.save_data_path, {'label': self.labels,
                                        'code_feat': self.data,
                                        'struct_feat': self.struct_feat,
                                        'mask': self.mask,
                                        'static_ids': self.static_ids,
                                        })

    def load(self):
        print("Loading data ...")
        info_dict = load_info(self.save_data_path)
        self.labels = info_dict['label']
        self.data = info_dict['code_feat']
        self.struct_feat = info_dict['struct_feat']
        self.mask = info_dict['mask']
        self.static_ids = info_dict['static_ids']

class CallGraphDatasetForInference(CallGraphDataset):
    def __init__(self, data_df: pd.DataFrame, save_data_path, mode, model_name="codebert", on_fly_process=False, disable_pb=False, progress_bar_pos=0):
        super().__init__(data_df, save_data_path, mode, model_name, on_fly_process, disable_pb, progress_bar_pos)

    def __getitem__(self, index):
        ids = self.data[index]
        mask = self.mask[index]
        return {
        'ids': np.array(ids, dtype=np.int64),
        'mask': np.array(mask, dtype=np.int64),
        'label': np.array(self.labels[index], dtype=np.int64),
        'static': np.array(self.static_ids[index], dtype=np.int64),
        }

def create_dataset(dataset_df, dataset_dir, dataset_name, model_name, pbar_pos, on_fly_process=False):
    if "struct" in dataset_name:
        return CallGraphDatasetWithStructFeat(dataset_df, dataset_dir, dataset_name)
    else:
        return CallGraphDataset(dataset_df, dataset_dir, dataset_name, model_name=model_name,
                                on_fly_process=on_fly_process, disable_pb=False, progress_bar_pos=pbar_pos)

def process_one_sample(src: str, dst: str) -> Dict[str, torch.Tensor]:
    token_ids, mask = get_input_and_mask(src, dst, MAX_SEQ_LENGTH, TOKENIZER)
    return {'ids': torch.tensor(token_ids, dtype=torch.long), 'mask': torch.tensor(mask, dtype=torch.long)}

def ignore_boot_methods(df_train:pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
    df_train = df_train[df_train['method'] != "<boot>"]
    df_valid = df_valid[df_valid['method'] != "<boot>"]
    df_test = df_test[df_test['method'] != "<boot>"]
    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)

def ignore_boot_methods_df(df: pd.DataFrame) -> pd.DataFrame:
     df = df[df['method'] != "<boot>"]
     return df

class DatasetTypes(Enum):
    # NJR 1
    ORG = 'org' # Original
    ORG_W_STRUCT = 'org_w_struct'
    ORG_ONE_CFA = 'org_1cfa_w_src'
    # END of NJR 1

    # XCorpus
    XCORP = 'xcorpus' # Xcorpus dataset
    XCORP_W_STRUCT = 'xcorpus_w_struct' # Xcorpus dataset w/ structural features
    XCORP_ONE_CFA = 'xcorpus_1cfa_w_src'
    # END of XCorpus

    # YCorpus: Dataset w/ high test coverage
    DHTC_W_SRC = 'dhtc_w_src'
    DHTC_W_STRUCT = 'dhtc_w_struct'
    DHTC_ONE_CFA = 'dhtc_1cfa_w_src'

    # NYXCorpus
    NYX_W_SRC = "nyx_w_src"
    NYX_W_STRUCT = "nyx_w_struct"
    NYX_ONE_CFA = "nyx_1cfa_w_src"


def load_dataset_files(dataset_name: DatasetTypes):
    NJR1_DATASET_PATH = join(PROJECT_DATA_PATH, "njr1")
    XCORPUS_DATASET = join(PROJECT_DATA_PATH, "xcorpus/dataset/")
    XCORPUS_DATASET_2 = join(PROJECT_DATA_PATH, "xcorpus")
    YCORPUS_DATASET = join(PROJECT_DATA_PATH, "ycorpus")
    NYX_DATASET_PATH = join(PROJECT_DATA_PATH, "nyx_dataset")
    train_path = None
    valid_path = None
    test_path = None
    dataset_path = None
    if dataset_name == DatasetTypes.ORG:
        train_path = join(NJR1_DATASET_PATH, "train_src2trg_org.csv")
        valid_path = join(NJR1_DATASET_PATH, "valid_src2trg_org.csv")
        test_path = join(NJR1_DATASET_PATH, "test_src2trg_org.csv")
        dataset_path = join(NJR1_DATASET_PATH)
    elif dataset_name == DatasetTypes.ORG_W_STRUCT:
        train_path = join(NJR1_DATASET_PATH, "train_src2trg_org_w_struct_feat.csv")
        valid_path = join(NJR1_DATASET_PATH, "valid_src2trg_org_w_struct_feat.csv")
        test_path = join(NJR1_DATASET_PATH, "test_src2trg_org_w_struct_feat.csv")
        dataset_path = join(NJR1_DATASET_PATH)
    elif dataset_name == DatasetTypes.ORG_ONE_CFA:
        train_path = join(NJR1_DATASET_PATH, "dataset/njr1_dataset_v1.0_w_src_train.csv")
        valid_path = join(NJR1_DATASET_PATH, "dataset/njr1_dataset_v1.0_w_src_valid.csv")
        test_path = join(NJR1_DATASET_PATH, "dataset/njr1_dataset_v1.0_w_src_test.csv")
        dataset_path = NJR1_DATASET_PATH
    elif dataset_name == DatasetTypes.XCORP:
        train_path = join(XCORPUS_DATASET, 'xcorpus_dataset_v4-3_w_src_train.csv')
        valid_path = join(XCORPUS_DATASET, 'xcorpus_dataset_v4-3_w_src_valid.csv')
        test_path = join(XCORPUS_DATASET, 'xcorpus_dataset_v4-3_w_src_test.csv')
        dataset_path = XCORPUS_DATASET
    elif dataset_name == DatasetTypes.XCORP_W_STRUCT:
        train_path = join(XCORPUS_DATASET, 'xcorpus_dataset_w_struct_v4-3_w_src_train.csv')
        valid_path = join(XCORPUS_DATASET, 'xcorpus_dataset_w_struct_v4-3_w_src_valid.csv')
        test_path = join(XCORPUS_DATASET, 'xcorpus_dataset_w_struct_v4-3_w_src_test.csv')
        dataset_path = XCORPUS_DATASET
    elif dataset_name == DatasetTypes.XCORP_ONE_CFA:
        train_path = join(XCORPUS_DATASET_2, 'xcorpus_dataset_v4-5_w_src_train.csv')
        valid_path = join(XCORPUS_DATASET_2, 'xcorpus_dataset_v4-5_w_src_valid.csv')
        test_path = join(XCORPUS_DATASET_2, 'xcorpus_dataset_v4-5_w_src_test.csv')
    elif dataset_name == DatasetTypes.DHTC_W_SRC:
        train_path = join(YCORPUS_DATASET, "high_tc_dataset_v7-3-5_w_src_train.csv")
        valid_path = join(YCORPUS_DATASET, "high_tc_dataset_v7-3-5_w_src_valid.csv")
        test_path = join(YCORPUS_DATASET, "high_tc_dataset_v7-3-5_w_src_test.csv")
        dataset_path = YCORPUS_DATASET
    elif dataset_name == DatasetTypes.DHTC_W_STRUCT:
        train_path = join(YCORPUS_DATASET, "high_tc_dataset_w_struct_v7-3-4_w_src_train.csv")
        valid_path = join(YCORPUS_DATASET, "high_tc_dataset_w_struct_v7-3-4_w_src_valid.csv")
        test_path = join(YCORPUS_DATASET, "high_tc_dataset_w_struct_v7-3-4_w_src_test.csv")
        dataset_path = YCORPUS_DATASET
    elif dataset_name == DatasetTypes.DHTC_ONE_CFA:
        train_path = join(YCORPUS_DATASET, "high_tc_dataset_v7-5_w_src_train.csv")
        valid_path = join(YCORPUS_DATASET, "high_tc_dataset_v7-5_w_src_valid.csv")
        test_path = join(YCORPUS_DATASET, "high_tc_dataset_v7-5_w_src_test.csv")
    elif dataset_name == DatasetTypes.NYX_W_SRC:
        train_path = join(NYX_DATASET_PATH, "nyx_dataset_train_w_src.csv")
        valid_path = join(NYX_DATASET_PATH, "nyx_dataset_valid_w_src.csv")
        test_path = join(NYX_DATASET_PATH, "nyx_dataset_test_w_src.csv")
        dataset_path = NYX_DATASET_PATH
    elif dataset_name == DatasetTypes.NYX_W_STRUCT:
        train_path = join(NYX_DATASET_PATH, "nyx_dataset_train_w_struct.csv")
        valid_path = join(NYX_DATASET_PATH, "nyx_dataset_valid_w_struct.csv")
        test_path = join(NYX_DATASET_PATH, "nyx_dataset_test_w_struct.csv")
        dataset_path = NYX_DATASET_PATH
    elif dataset_name == DatasetTypes.NYX_ONE_CFA:
        train_path = join(NYX_DATASET_PATH, "nyx_dataset_train_1cfa_w_src.csv")
        valid_path = join(NYX_DATASET_PATH, "nyx_dataset_valid_1cfa_w_src.csv")
        test_path = join(NYX_DATASET_PATH, "nyx_dataset_test_1cfa_w_src.csv")
        dataset_path = NYX_DATASET_PATH
    
    return train_path, valid_path, test_path, dataset_path

def get_X_Y_arrays(dataset_path: str, df: pd.DataFrame, dataset_version=""):
    X_list, Y_list, Y_wala = [], [], []
    programs_list = []
    df['m2t'] = df.apply(lambda x: x['method']+"|"+x['target'], axis=1)
    all_proj_m2t = []
    all_proj_df = []
    for p in tqdm(df['program_name'].unique()):
        if 'xcorpus' in dataset_path:
            proj_df = pd.read_csv(join(dataset_path, p + f"_w_edge_feat_{dataset_version}.csv"))
        elif 'ycorpus' in dataset_path:
            proj_df = pd.read_csv(join(dataset_path, p.replace("/", "-") + f"_w_edge_feat_{dataset_version}.csv"))
        else:
            proj_df = pd.read_csv(join(dataset_path, p, 'wala0cfa.csv'))
        proj_df = proj_df[proj_df['method'] != "<boot>"]
        proj_df['m2t'] = proj_df.apply(lambda x: x['method']+"|"+x['target'], axis=1)
        proj_df['m2t_w_off'] = proj_df.apply(lambda x: x['method']+"|"+str(x['offset'])+"|"+x['target'], axis=1)
        proj_df = proj_df[proj_df['m2t'].isin(df['m2t'])]
        proj_df.drop_duplicates(subset='m2t_w_off', keep='first', inplace=True)
        all_proj_m2t = all_proj_m2t + proj_df['m2t'].to_list()
        proj_df.drop('m2t', axis=1, inplace=True)
        proj_df.drop('m2t_w_off', axis=1, inplace=True)
        all_proj_df.append(proj_df)
        proj_df_y = proj_df['wiretap']
        proj_df_wala = proj_df['wala-cge-0cfa-noreflect-intf-trans']
        wala_direct_cols = [col for col in proj_df.columns if col.startswith('wala-cge-0cfa-noreflect-intf-direct#')]
        proj_df_x = proj_df.loc[:, ~proj_df.columns.isin(['wiretap', 'wala-cge-0cfa-noreflect-intf-trans',
                                                            'wala-cge-0cfa-noreflect-intf-direct',
                                                            'method', 'offset', 'target'] + \
                                                            wala_direct_cols + FEATURES_TO_REMOVE)]
        programs_list = programs_list + [p] * proj_df_y.shape[0]

        X_list.append(proj_df_x.to_numpy())
        Y_list.append(proj_df_y.to_numpy())
        Y_wala.append(proj_df_wala.to_numpy())

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    Y_wala = np.concatenate(Y_wala)
    all_proj_df = pd.concat(all_proj_df, ignore_index=True)

    return X, Y, Y_wala, programs_list, all_proj_m2t, all_proj_df

def tokenize_and_count(text: str):
    return len(TOKENIZER(text)['input_ids'])

def get_token_size_s2t(df, col_method, col_target) -> int:
    df = df[df['method'] != "<boot>"]
    with Pool(processes=2) as pool:
        method_token_size = sum(pool.map(tokenize_and_count, df[col_method]))
        target_token_size = sum(pool.map(tokenize_and_count, df[col_target]))
    return method_token_size + target_token_size

def get_call_graphs_from_test_data(df_test: pd.DataFrame, edge_label: str) -> dict:
    test_progs_cg = {}
    test_progs = df_test['program_name'].unique()
    for p in tqdm(test_progs):
        test_progs_cg[p] = {'nodes': [], 'edges': []}
        for i, r in df_test[df_test['program_name'] == p].iterrows():
            if r[edge_label] == 1:
                test_progs_cg[p]['edges'].append((r['method'], r['target'])) # s -> t
        test_progs_cg[p]['nodes'] = set([n for e in test_progs_cg[p]['edges'] for n in e])
    return test_progs
