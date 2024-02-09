"""
A set of utilties to for processing datasets
"""

import re
import pandas as pd
from src.converter import convert
from tqdm import tqdm

def report_nodes_w_src(df_dataset: pd.DataFrame) -> pd.DataFrame:
    def detect_no_src(func: str):
        func = str(func)
        if func == '':
            return True
        elif bool(re.search("^class.*; }$", func)):
            return True
        return False
              
    m_lq = df_dataset['method_src'].apply(detect_no_src)
    t_lq = df_dataset['target_src'].apply(detect_no_src)
    return df_dataset[~m_lq & ~t_lq]

def has_node_src_code(node_src: str) -> bool:
    """
    Checks whether a CG node/method has full source code, not an artificial one
    """
    return not bool(re.search("^class.*; }$", node_src))

def add_artificial_code(df):
    for i, r in tqdm(df.iterrows(), total=len(df)):
        try:
            if r['method'] != "<boot>":
                df.at[i, 'method_src'] = convert(r['method']).__tocode__()
                df.at[i, 'target_src'] = convert(r['target']).__tocode__()
            else:
                df.at[i, 'method_src'] = ""
                df.at[i, 'target_src'] = convert(r['target']).__tocode__()
        except IndexError as e:
            print(r['target'])
    return df
