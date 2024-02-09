"""
This script creates random vulnerable CG nodes or fake CVEs.
"""

from src import PROJECT_DATA_PATH
from os.path import join
import pandas as pd
import networkx as nx
import random
import json

RESULTS_FOLDER = join(PROJECT_DATA_PATH, "results")
OUTPUT_FOLDER = join(PROJECT_DATA_PATH, "nyx_dataset")
NUM_RAND_VULN_NODES = 100
PRUNE_CONFIDENCE_THRESHOLD = 0.95

if __name__ == "__main__":
    cgs = {"codebert": {},
           "codebert_C99": {},
           "codet5": {},
           "codet5_C99": {},
           "RC": {},
           "wala": {}}
    cgs_nx_graph = {"codebert": {},
                    "codebert_C99": {},
                    "codet5": {},
                    "codet5_C99": {},
                    "RC": {},
                    "wala": {}}
    # wala_source_nodes = {}
    # wala_vuln_nodes = {}
    for m in ['codebert', 'codebert_C99', 'codet5', 'codet5_C99', 'RC']:
        # 0-CFA
        if m != "codebert_C99" and m != 'codet5_C99':
            df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{m}_pruner_now_nyx_w_src.csv"))
        elif m == "codebert_C99":
            df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_codebert_pruner_C099_nyx_w_src.csv"))
        elif m == "codet5_C99":
            df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_codet5_pruner_C099_nyx_w_src.csv"))

        for p in df_res['program_name'].unique():
            df_res_p = df_res[df_res['program_name'] == p].copy()
            df_res_p = df_res_p[df_res_p['wala-cge-0cfa-noreflect-intf-trans'] == 1]
            # Ignore dynamic edges not present in the static CGs
            #df_res_p = df_res_p[~((df_res_p['wiretap'] == 1) & (df_res_p['wala-cge-0cfa-noreflect-intf-trans'] == 0) & (df_res_p['m_out'] == 1))]

            cgs[m][p] = {'sourceNodes': None, 'cg': None}
            cgs["wala"][p] = {'sourceNodes': None, 'vulnerableNodes': None, 'cg': None}
            df_res_p_wala = df_res_p[df_res_p['wala-cge-0cfa-noreflect-intf-trans'] == 1]
            if m != "RC":
                df_res_p['m_out_p_prune'] = df_res_p['m_out_p'].apply(lambda x : 1 - x)
                df_res_p['m_out'] = df_res_p['m_out_p_prune'].apply(lambda x: 0 if x > PRUNE_CONFIDENCE_THRESHOLD else 1)
            df_res_p_pruned = df_res_p[df_res_p['m_out'] == 1]
            cgs[m][p]['cg'] = list(zip(df_res_p_pruned['method'], df_res_p_pruned["target"]))
            m_g_p = nx.DiGraph()
            for s, t in cgs[m][p]['cg']:
                m_g_p.add_edge(s, t) 
            cgs[m][p]['sourceNodes'] = [node for node, in_degree in m_g_p.in_degree() if in_degree == 0]
            # WALA
            if cgs["wala"][p]['cg'] is None:
                cgs["wala"][p]['cg'] = list(zip(df_res_p_wala['method'], df_res_p_wala["target"]))
                wala_g_p = nx.DiGraph()
                for s, t in cgs["wala"][p]['cg']:
                    wala_g_p.add_edge(s, t)
                cgs_nx_graph["wala"] = wala_g_p
                cgs["wala"][p]['sourceNodes'] = [node for node, in_degree in wala_g_p.in_degree() if in_degree == 0]
                cgs["wala"][p]['vulnerableNodes'] = random.sample([node for node, in_degree in wala_g_p.in_degree() if in_degree != 0], NUM_RAND_VULN_NODES)
                print(f"[{p}] No. of source nodes for WALA's static CG {len(cgs['wala'][p]['sourceNodes']):,}")
                print(f"[{p}] No. of edges for WALA's static CG {len(df_res_p_wala):,}")
            
            print(f"[{p}] No. of edges for {m}'s pruned CG: {len(df_res_p_pruned):,}" )
    
    with open(join(OUTPUT_FOLDER, "nyx_corpus_cgs_w_vuln_nodes.json"), "w") as f:
        json.dump(cgs, f, indent=4)