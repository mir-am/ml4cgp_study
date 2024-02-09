"""
This creates a "meta" dataset, NYXCorpus, from NJR-1, XCorpus, and YCorpus
"""

from src.dataset import DatasetTypes, load_dataset_files
from src import PROJECT_DATA_PATH
from os.path import join
import pandas as pd

OUTFOLDER = join(PROJECT_DATA_PATH, "/nyx_dataset")

def load_dataset_df(train_path, valid_path, test_path):
      df_train = pd.read_csv(train_path)
      df_valid = pd.read_csv(valid_path)
      df_test = pd.read_csv(test_path)
      return df_train, df_valid, df_test

def create_nyx_corpus_w_struct_0cfa():
    #NJR-1
    njr1_train_path, njr1_valid_path, njr1_test_path, _ = load_dataset_files(DatasetTypes.ORG_W_STRUCT)
    njr1_train_df, njr1_valid_df, njr1_test_df = load_dataset_df(njr1_train_path, njr1_valid_path, njr1_test_path)

    #XCorpus
    xcorp_train_path, xcorp_valid_path, xcorp_test_path, _ = load_dataset_files(DatasetTypes.XCORP_W_STRUCT)
    xcorp_train_df, xcorp_valid_df, xcorp_test_df = load_dataset_df(xcorp_train_path, xcorp_valid_path, xcorp_test_path)

    #YCorpus
    ycorp_train_path, ycorp_valid_path, ycorp_test_path, _ = load_dataset_files(DatasetTypes.DHTC_W_STRUCT)
    ycorp_train_df, ycorp_valid_df, ycorp_test_df = load_dataset_df(ycorp_train_path, ycorp_valid_path, ycorp_test_path)

    metadataset_train_df = pd.concat([njr1_train_df, xcorp_train_df ,ycorp_train_df], ignore_index=True)
    metadataset_valid_df = pd.concat([njr1_valid_df, xcorp_valid_df ,ycorp_valid_df], ignore_index=True)       
    metadataset_test_df = pd.concat([njr1_test_df, xcorp_test_df, ycorp_test_df], ignore_index=True)

    print(f"Metadata dataset with train {metadataset_train_df.shape} valid {metadataset_valid_df.shape} test {metadataset_test_df.shape}")

    metadataset_train_df.to_csv(join(OUTFOLDER, "nyx_dataset_train_w_struct.csv"), index=False)
    metadataset_valid_df.to_csv(join(OUTFOLDER, "nyx_dataset_valid_w_struct.csv"), index=False)
    metadataset_test_df.to_csv(join(OUTFOLDER, "nyx_dataset_test_w_struct.csv"), index=False)

def create_nyx_corpus_w_src_0cfa():
    #NJR-1
    njr1_train_path, njr1_valid_path, njr1_test_path, _ = load_dataset_files(DatasetTypes.ORG)
    njr1_train_df, njr1_valid_df, njr1_test_df = load_dataset_df(njr1_train_path, njr1_valid_path, njr1_test_path)

    #XCorpus
    xcorp_train_path, xcorp_valid_path, xcorp_test_path, _ = load_dataset_files(DatasetTypes.XCORP)
    xcorp_train_df, xcorp_valid_df, xcorp_test_df = load_dataset_df(xcorp_train_path, xcorp_valid_path, xcorp_test_path)

    #YCorpus
    ycorp_train_path, ycorp_valid_path, ycorp_test_path, _ = load_dataset_files(DatasetTypes.DHTC_W_SRC)
    ycorp_train_df, ycorp_valid_df, ycorp_test_df = load_dataset_df(ycorp_train_path, ycorp_valid_path, ycorp_test_path)

    metadataset_train_df = pd.concat([njr1_train_df, xcorp_train_df ,ycorp_train_df], ignore_index=True)
    metadataset_valid_df = pd.concat([njr1_valid_df, xcorp_valid_df ,ycorp_valid_df], ignore_index=True)       
    metadataset_test_df = pd.concat([njr1_test_df, xcorp_test_df, ycorp_test_df], ignore_index=True)

    print(f"Metadata dataset with train {metadataset_train_df.shape} valid {metadataset_valid_df.shape} test {metadataset_test_df.shape}")

    metadataset_train_df.to_csv(join(OUTFOLDER, "nyx_dataset_train_w_src.csv"), index=False)
    metadataset_valid_df.to_csv(join(OUTFOLDER, "nyx_dataset_valid_w_src.csv"), index=False)
    metadataset_test_df.to_csv(join(OUTFOLDER, "nyx_dataset_test_w_src.csv"), index=False)

def create_nyx_corpus_w_src_1cfa():
    #NJR-1
    njr1_train_path, njr1_valid_path, njr1_test_path, _ = load_dataset_files(DatasetTypes.ORG_ONE_CFA)
    njr1_train_df, njr1_valid_df, njr1_test_df = load_dataset_df(njr1_train_path, njr1_valid_path, njr1_test_path)

    #XCorpus
    xcorp_train_path, xcorp_valid_path, xcorp_test_path, _ = load_dataset_files(DatasetTypes.XCORP_ONE_CFA)
    xcorp_train_df, xcorp_valid_df, xcorp_test_df = load_dataset_df(xcorp_train_path, xcorp_valid_path, xcorp_test_path)

    #YCorpus
    ycorp_train_path, ycorp_valid_path, ycorp_test_path, _ = load_dataset_files(DatasetTypes.DHTC_ONE_CFA)
    ycorp_train_df, ycorp_valid_df, ycorp_test_df = load_dataset_df(ycorp_train_path, ycorp_valid_path, ycorp_test_path)

    metadataset_train_df = pd.concat([njr1_train_df, xcorp_train_df ,ycorp_train_df], ignore_index=True)
    metadataset_valid_df = pd.concat([njr1_valid_df, xcorp_valid_df ,ycorp_valid_df], ignore_index=True)       
    metadataset_test_df = pd.concat([njr1_test_df, xcorp_test_df, ycorp_test_df], ignore_index=True)

    print(f"Metadata dataset with train {metadataset_train_df.shape} valid {metadataset_valid_df.shape} test {metadataset_test_df.shape}")

    metadataset_train_df.to_csv(join(OUTFOLDER, "nyx_dataset_train_1cfa_w_src.csv"), index=False)
    metadataset_valid_df.to_csv(join(OUTFOLDER, "nyx_dataset_valid_1cfa_w_src.csv"), index=False)
    metadataset_test_df.to_csv(join(OUTFOLDER, "nyx_dataset_test_1cfa_w_src.csv"), index=False)

if __name__ == "__main__":
    create_nyx_corpus_w_src_0cfa()
    create_nyx_corpus_w_struct_0cfa()
    create_nyx_corpus_w_src_1cfa()
