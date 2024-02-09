from src import utils
from src import eval
from src.constants import struct_feat_names, PREFIXES_TO_REMOVE
from src.dataset_utils import add_artificial_code, has_node_src_code, report_nodes_w_src
from src.utils import report_training_samples_types
from src.feature_ext import UnionEdge, Graph, Node, Edge, \
compute_node_and_edge_counts, get_orphan_nodes, compute_edge_depths, compute_edge_reachability, \
compute_src_node_in_deg, compute_dest_node_in_deg, compute_edge_fanouts, compute_repeated_edges, \
compute_node_disjoint_paths, compute_edge_disjoint_paths, compute_graph_level_info, remove_repeated_edges_from_union, \
compute_output_imp, add_old_entries_to_row_imp, compute_edge_disjoint_paths_parallel
from src import PROJECT_DATA_PATH

import pandas as pd
import os
import glob
import math
import psutil
import zipfile
import subprocess
import faulthandler
from os.path import join, isdir, splitext, dirname, relpath, exists
from enum import Enum
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from os import listdir, walk
from typing import Dict, Tuple
from tqdm import tqdm
faulthandler.enable()

SOURCE_RECOVER_JAR = "/mnt/data/amir_projects/ml4cg/ml4cg_SA/target/method_extractor-1.0-SNAPSHOT-shaded.jar"
NO_WORKERS = math.floor(0.9 * psutil.cpu_count(logical=False))

## Read dataset
def load_ycorpus_data(dataset_folder, jars_folder, limit_projects=-1, use_1cfa=False):
    project_static_cg = {}
    project_dynamic_cg = {}
    project_cg_jars = {}
    project_static_cg_f = {}
    project_dynamic_cg_f = {}

    for root, dirs, files in os.walk(dataset_folder):
        project_n = os.path.relpath(root, dataset_folder)
        for d in dirs:
            if '_wiretap' in d:
                if len(glob.glob(join(root, "*SNAPSHOT.csv"))) != 0:
                    project_dyn_cg_f = join(root, d, "merged_test-classes_dyn_cg.csv")
                    if os.path.getsize(project_dyn_cg_f) > 0:
                        project_dyn_cg_df = pd.read_csv(project_dyn_cg_f, on_bad_lines='skip', names=['source', 'target'])
                        if len(project_dyn_cg_df) > 100:
                            project_dynamic_cg[project_n] = project_dyn_cg_df
                            project_dynamic_cg_f[project_n] = join(project_dyn_cg_f)
        for f in files:
            if exists(join(root, "_wiretap")):
                if f.endswith(f"SNAPSHOT{'_0cfa' if use_1cfa else ''}.csv"):
                    try:
                        if project_n not in project_static_cg:
                            project_static_cg_df = pd.read_csv(join(root, f))
                            if len(project_static_cg_df) > 100:
                                project_static_cg[project_n] = project_static_cg_df
                                project_static_cg_f[project_n] = join(root, f)
                        else:
                            project_static_cg[project_n] = pd.concat([project_static_cg[project_n],
                                                                  pd.read_csv(join(root, f))]).reset_index(drop=True)
                    except pd.errors.EmptyDataError:
                        print(f"Empty data {f}")
                    jar_f_n = join(jars_folder, project_n, f.replace("_0cfa", "").replace(".csv", ".jar"))
                    if exists(jar_f_n):
                        project_cg_jars[project_n] = jar_f_n
                    else:
                        raise RuntimeError(jar_f_n)
     
    edges_no = 0
    for k, v in project_static_cg.items():
        edges_no += v.shape[0]
    print(f"Total no. of edges in the dataset {edges_no}")
    return project_static_cg, project_dynamic_cg, project_cg_jars

def load_xcorpus_data(dataset_folder, use_1cfa=False):
    projects_xcorpus = [(p, 'qualitas_corpus_20130901') for p in listdir(join(dataset_folder, 'qualitas_corpus_20130901'))] + \
                       [(p, 'xcorpus-extension-20170313') for p in listdir(join(dataset_folder, 'xcorpus-extension-20170313'))]
    project_static_cg = {}
    project_dynamic_cg: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    project_static_cg_f = {}
    project_dynamic_cg_f = {}
    # projects_xcorpus = [(p, 'qualitas_corpus_20130901') for p in listdir(join(DATASET_FOLDER, 'qualitas_corpus_20130901'))] + \
    #                    [(p, 'xcorpus-extension-20170313') for p in listdir(join(DATASET_FOLDER, 'xcorpus-extension-20170313'))]
    for project_name, dataset in tqdm(projects_xcorpus):
        project_folder = join(dataset_folder, dataset, project_name)
        if isdir(project_folder):
            for file_name in listdir(project_folder):
                if file_name == "_wiretap":
                    try:
                        dyn_cg_bt = pd.read_csv(join(project_folder, file_name, "merged_builtin-tests_dyn_cg.csv"),
                                                names=['source', 'target'])
                        dyn_cg_gt = pd.read_csv(join(project_folder, file_name, "merged_generated-tests_dyn_cg.csv"),
                                                names=['source', 'target'])
                        dyn_cg_bt['program_name'] = project_name
                        dyn_cg_gt['program_name'] = project_name
                        project_dynamic_cg[project_name] = pd.concat((dyn_cg_bt, dyn_cg_gt))
                        combined_dyn_cg_path = join(project_folder, file_name, "merged_combined-tests_dyn_cg.csv")
                        project_dynamic_cg_f[project_name] = combined_dyn_cg_path
                        if not exists(join(combined_dyn_cg_path)):
                            project_dynamic_cg[project_name].to_csv(combined_dyn_cg_path, index=False)
                    except FileNotFoundError:
                        pass
                if file_name.lower().endswith(f'{"_1cfa" if use_1cfa else ""}.csv'):
                    csv_file_path = join(project_folder, file_name)
                    try:
                        project_static_cg[project_name] = pd.read_csv(csv_file_path)
                        project_static_cg_f[project_name] = csv_file_path
                    except pd.errors.EmptyDataError:
                        pass
    return project_static_cg, project_dynamic_cg

def get_no_static_dyn_edges(static_dyn_common, project_static_cg, project_dynamic_cg):
        projects_common = {}
        for p in static_dyn_common:
            main_p = "/".join(p.split("/")[:2])
            if main_p not in projects_common:
                projects_common[main_p] = {'static': project_static_cg[p].shape[0], 'dynamic': project_dynamic_cg[p].shape[0]}
            else:
                projects_common[main_p]['static'] += project_static_cg[p].shape[0]
                projects_common[main_p]['dynamic'] += project_dynamic_cg[p].shape[0]
        return projects_common

def find_common_projects(project_static_cg, project_dynamic_cg):
    static_dyn_common = set(project_static_cg.keys()).intersection(set(project_dynamic_cg))

    for p in static_dyn_common:
        print(f"Project {p} has {project_static_cg[p].shape[0]:,} static edges and has {project_dynamic_cg[p].shape[0]:,} dynamic edges")

    projects_common = get_no_static_dyn_edges(set(project_static_cg.keys()).intersection(set(project_dynamic_cg.keys())),
                                          project_static_cg, project_dynamic_cg)
    print(f"No. of projects {len(projects_common)}")
    return static_dyn_common, projects_common
# project_static_cg_pruned = project_static_cg
# project_dynamic_cg_pruned = project_dynamic_cg

def extract_jar_and_find_class_files(jar_path, dest_path):
    with zipfile.ZipFile(jar_path, 'r') as jar:
        jar.extractall(path=dest_path)

    class_files = []
    for dirpath, dirs, files in os.walk(dest_path):
        for file in files:
            if file.endswith('.class'):
                class_files.append(os.path.join(dirpath, file))
                
    return class_files

def get_ns_based_cgs(project_static_cg, project_dynamic_cg, project_cg_jars, static_dyn_common):
    def process_row(r, project_ns, tgt_ns_set):
        src_ns, tgt_ns = utils.extract_ns_from_uri(r['source']), utils.extract_ns_from_uri(r['target'])
        if src_ns in project_ns or tgt_ns in project_ns:
            if tgt_ns not in project_ns:
                tgt_ns_set.add(tgt_ns)
            return True
        if src_ns in tgt_ns_set or tgt_ns in tgt_ns_set:
            return True
        return False

    project_cg_ns = {}
    for p, j in tqdm(project_cg_jars.items()):
        if p in project_static_cg:
            p_name = os.path.basename(j).replace(".jar", "")
            dest_path = join("/tmp/", p_name)
            cls_files = extract_jar_and_find_class_files(j, dest_path)
            project_cg_ns[p] = set([os.path.dirname(os.path.relpath(c_f, dest_path)) for c_f in cls_files])

    project_dynamic_cg_pruned_ns = {}
    for p in tqdm(static_dyn_common):
        p_df = project_dynamic_cg[p].reset_index()
        p_tgt_ns = set()
    
        p_df['relevant'] = p_df.apply(process_row, axis=1, args=(project_cg_ns[p], p_tgt_ns))
        project_dynamic_cg_pruned_ns[p] = set(p_df[p_df['relevant']].index)
        print(f"No. of samples for {p} reduced from {p_df.shape[0]} to {len(project_dynamic_cg_pruned_ns[p])}")


    project_static_cg_pruned_ns = {}
    for p in tqdm(static_dyn_common):
        p_df = project_static_cg[p].reset_index()
        p_tgt_ns = set()
    
        p_df['relevant'] = p_df.apply(process_row, axis=1, args=(project_cg_ns[p], p_tgt_ns))
        project_static_cg_pruned_ns[p] = set(p_df[p_df['relevant']].index)
        print(f"No. of samples for {p} reduced from {p_df.shape[0]} to {len(project_static_cg_pruned_ns[p])}")

    for p in tqdm(static_dyn_common):
        p_df_dyn = project_dynamic_cg[p]
        p_df_static = project_static_cg[p]
        project_dynamic_cg[p] = p_df_dyn.iloc[list(project_dynamic_cg_pruned_ns[p])]
        project_static_cg[p] = p_df_static.iloc[list(project_static_cg_pruned_ns[p])]

def process_p(p):
        prefixes_to_remove = PREFIXES_TO_REMOVE
        prefixes_to_remove = tuple(prefixes_to_remove)

        m_bt_gt = project_dynamic_cg[p][['source', 'target']].drop_duplicates()
        m_idx_bt_gt = {f"{r.source}|{r.target}": i for i, r in m_bt_gt.iterrows()}
        m_idx_bt_st = {f"{r.source}|{r.target}": i for i, r in project_static_cg[p].iterrows()}
        r_static_processed = set()
        r_static_nodes_visited = set()
    #r_dyn_processed = set()

        def process_row_static(r):
            source_target = f"{r.source}|{r.target}"
            if source_target not in r_static_processed:
                r_static_processed.add(source_target)
                r_static_nodes_visited.add(r.source)
                r_static_nodes_visited.add(r.target)
            else:
                return None
        
            if r.target == "com/ibm/wala/FakeRootClass.fakeWorldClinit:()V" or \
                r.source.startswith(prefixes_to_remove) or r.target.startswith(prefixes_to_remove):
                return None
       
            if source_target in m_idx_bt_gt:
                return [1, 1, r.source, r.offset, r.target, r.source_ln, r.target_ln, "", ""]
            else:
                return [0, 1, r.source, r.offset, r.target, r.source_ln, r.target_ln, "", ""]

        def process_row_dynamic(r):
            source_target = f"{r.source}|{r.target}"
            if r.source not in r_static_nodes_visited and r.target not in r_static_nodes_visited:
                return None
            if "Test" in utils.extract_class_name(r.source) or "Test" in utils.extract_class_name(r.target) or \
                r.source.startswith(prefixes_to_remove) or r.target.startswith(prefixes_to_remove):
                return None
            if source_target not in m_idx_bt_st:
                return [1, 0, r.source, -1, r.target, "-1 -1", "-1 -1", "", ""]

        dataset_rows_static = project_static_cg[p].apply(process_row_static, axis=1).dropna().tolist()
        dataset_rows_dynamic = m_bt_gt.apply(process_row_dynamic, axis=1).dropna().tolist()
        dataset_rows = dataset_rows_static + dataset_rows_dynamic
        return pd.DataFrame(dataset_rows,
                        columns=['wiretap', 'wala-cge-0cfa-noreflect-intf-trans', 'method', 'offset',
                                'target', 'method_ln', 'target_ln', 'method_src', 'target_src'])

def process_project_struct_feat(proj, output_folder, dataset_version):
    main_method = "<boot>"
    #cg_df = static_dyn_common[proj]
    cg_df = concat_dataset[concat_dataset['program_name'] == proj]

    union_edge_set = []
    call_graphs_w_feat = Graph()

    for i, r in cg_df.iterrows():
        union_edge_set.append(UnionEdge(r['method'], r['offset'], r['target'], r['wiretap'],
                                        r['wala-cge-0cfa-noreflect-intf-trans']))
        if r['method'] not in call_graphs_w_feat.nodes:
            call_graphs_w_feat.nodes[r['method']] = Node()
        if r['target'] not in call_graphs_w_feat.nodes:
            call_graphs_w_feat.nodes[r['target']] = Node()
        call_graphs_w_feat.nodes[r['method']].edges.add(Edge(r['offset'], r['target']))

    compute_node_and_edge_counts(call_graphs_w_feat)
    orphan_nodes = get_orphan_nodes(call_graphs_w_feat)

    if main_method in orphan_nodes:
        orphan_nodes.remove(main_method)

    compute_edge_depths(call_graphs_w_feat, main_method, orphan_nodes)
    print(f"Computed edge depths for project: {proj}")
    compute_edge_reachability(call_graphs_w_feat)
    print(f"Computed edge reachability for project: {proj}")
    compute_src_node_in_deg(call_graphs_w_feat)
    print(f"Computed source node in-degrees for project: {proj}")
    compute_dest_node_in_deg(call_graphs_w_feat)
    print(f"Computed destination node in-degrees for project: {proj}")
    compute_edge_fanouts(call_graphs_w_feat)
    print(f"Computed edge fanouts for project: {proj}")
    compute_repeated_edges(call_graphs_w_feat)
    print(f"Computed repeated edges for project: {proj}")
    compute_node_disjoint_paths(call_graphs_w_feat, main_method, orphan_nodes)
    print(f"Computed node disjoint paths for project: {proj}")
    compute_edge_disjoint_paths_parallel(call_graphs_w_feat, main_method, orphan_nodes)
    print(f"Computed edge disjoint paths for project: {proj}")
    compute_graph_level_info(call_graphs_w_feat, orphan_nodes)
    print(f"Computed graph level information for project: {proj}")
    remove_repeated_edges_from_union(union_edge_set)
    print(f"Removed repeated edges for project: {proj}")

    edge_samples = []
    for union_edge in union_edge_set:
        row_sample = {}
        add_old_entries_to_row_imp(row_sample, union_edge)
        compute_output_imp(row_sample, union_edge, call_graphs_w_feat,
                                          'wala-cge-0cfa-noreflect-intf-trans')
        edge_samples.append(row_sample)
    edge_samples_df = pd.DataFrame.from_dict(edge_samples)
    edge_samples_df.to_csv(join(output_folder, proj.replace("/", "-") + f"_w_edge_feat_{dataset_version}.csv"), index=False)
    return proj, edge_samples_df

def gen_struct_features(static_dyn_common, output_folder, dataset_version):
    projects_w_feat = {}
    for proj in tqdm(static_dyn_common):
        proj, df = process_project_struct_feat(proj, output_folder, dataset_version)
        projects_w_feat[proj] = df

    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(process_project_struct_feat, proj, output_folder, dataset_version) for proj in tqdm(static_dyn_common)]
    #     for future in futures:
    #         proj, df = future.result()
    #         projects_w_feat[proj] = df

def add_struct_features(static_dyn_common, concat_dataset_df, output_folder, dataset_version):
    dataset_programs_feat = {}
    dataset_programs_df_idx = {}
    def add_struct_feat_to_df(df: pd.DataFrame):
        for i, r in tqdm(df.iterrows(), total=len(df), desc="Add struct feat"):
            dataset_prog_df = dataset_programs_feat[r['program_name']]
            match_r = dataset_programs_df_idx[r['program_name']][r['method']+"|"+r['target']]
            for f in struct_feat_names:
                df.at[i, f] = dataset_prog_df.iloc[match_r][f]
        return df

    for p in tqdm(static_dyn_common):
        proj_df = pd.read_csv(join(output_folder, p.replace("/", "-") + f"_w_edge_feat_{dataset_version}.csv"))
        proj_df['program_name'] = p
        dataset_programs_feat[p] = proj_df
        dataset_programs_df_idx[p] = {}
        for i, r in proj_df.iterrows():
            dataset_programs_df_idx[p][r['method']+"|"+r['target']] = i
    
    for f in struct_feat_names:
        concat_dataset_df[f] = 0.0

    add_struct_feat_to_df(concat_dataset)
    scaler = MinMaxScaler()
    for f in struct_feat_names:
        concat_dataset[f] = scaler.fit_transform(concat_dataset[f].to_numpy().reshape(-1, 1))

    return concat_dataset_df

def create_dataset(output_folder, dataset_version, project_static_cg, project_dynamic_cg):
    static_dyn_common_pruned = set(project_static_cg.keys()).intersection(set(project_dynamic_cg.keys()))

    # Create a dataset from  programs
    with Pool(NO_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_p, static_dyn_common_pruned), total=len(static_dyn_common_pruned)))

    static_dyn_common_pruned_df = {p: df for p, df in zip(static_dyn_common_pruned, results)}
    for p, v in static_dyn_common_pruned_df.items():
        p_no_st_edges = v[v['wala-cge-0cfa-noreflect-intf-trans'] == 1].shape[0]
        p_no_dyn_edges = v[v['wiretap'] == 1].shape[0]
        print(f"Project {p} has {p_no_st_edges:,} static edges and has {p_no_dyn_edges:,} dynamic edges")

    for p in static_dyn_common_pruned_df:
        static_dyn_common_pruned_df[p].to_csv(join(output_folder, p.replace("/", "_")+f"_dataset_{dataset_version}.csv"), index=False)
    return static_dyn_common_pruned_df

def concat_datasets(output_folder, dataset_name, static_dyn_common_pruned_df):
    high_tc_data_df = pd.concat([df.assign(program_name=name) for name, df in static_dyn_common_pruned_df.items()])
    high_tc_data_df.to_csv(join(output_folder, f"{dataset_name}.csv"), index=False)

    utils.report_training_samples_types(high_tc_data_df)
    eval.compute_eval_metrics_wala(high_tc_data_df)
    
    # Dataset sampling
    high_tc_data_df_sampled = high_tc_data_df.copy().reset_index(drop=True)
    # max_samples = 20000
    # high_tc_data_df_sampled = high_tc_data_df.groupby('program_name').apply(lambda x: x.sample(min(len(x),
    #                                                                                            max_samples))).reset_index(drop=True)
    max_samples = 20000
    filtered_df = high_tc_data_df_sampled[~((high_tc_data_df_sampled['wiretap'] == 1) & (high_tc_data_df_sampled['wala-cge-0cfa-noreflect-intf-trans'] == 1))]
    sampled_rows = filtered_df.groupby('program_name').apply(lambda x: x.sample(min(len(x), max_samples)))
    high_tc_data_df_sampled = high_tc_data_df_sampled.drop(index=filtered_df.index)
    high_tc_data_df_sampled= pd.concat([high_tc_data_df_sampled, sampled_rows])
    high_tc_data_df_sampled = high_tc_data_df_sampled.reset_index(drop=True)

    utils.report_training_samples_types(high_tc_data_df_sampled)
    eval.compute_eval_metrics_wala(high_tc_data_df_sampled)

    #high_tc_data_df_sampled.to_csv(join(output_folder, f'{dataset_name}_sampled.csv'), index=False)

    # Add sig-based representation
    high_tc_data_df_sampled = add_artificial_code(high_tc_data_df_sampled)
    high_tc_data_df_sampled.to_csv(join(output_folder, f'{dataset_name}_sampled.csv'), index=False)

    return high_tc_data_df_sampled


def recover_dataset_source(output_folder, source_folder, dataset_name, second_source_folder=None):
    # Recover source code
    ghp_dataset_folder = join(output_folder, f'{dataset_name}_sampled.csv')
    ghp_output_file = join(output_folder, f'{dataset_name}_sampled_w_src.csv')
    second_source_opt = f'-s2 {second_source_folder}' if second_source_folder else ''

    command = f'java -cp {SOURCE_RECOVER_JAR} dev.c0pslab.analysis.SourceAdder -s {source_folder} {second_source_opt} -d {ghp_dataset_folder} -o {ghp_output_file}'
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode == 0:
        print(f"Recovered source code successfully.")
    else:
        raise RuntimeError("Failed to recover source code.")

    concat_dataset_recoverd = pd.read_csv(join(output_folder, f'{dataset_name}_sampled_w_src.csv'))
    return concat_dataset_recoverd

def save_dataset_w_src(concat_dataset_df, output_folder, dataset_name, dataset_split=None):
    # Load dataset with source code recovered
    #high_tc_data_df_sampled = pd.read_csv(join(output_folder, f'{dataset_name}_sampled_w_src.csv'))
    high_tc_data_df_sampled = concat_dataset_df
    if dataset_split is None:
        if "high_tc_dataset" in dataset_name:
            dataset_split = {'train': ['zalando/problem-spring-web/problem-spring-webflux',
                            'zalando/problem-spring-web/problem-spring-common',
                            'zalando/problem/problem',
                            'zalando/problem-spring-web/problem-violations',
                            'zalando/problem-spring-web/problem-spring-web',
                            'fridujo/rabbitmq-mock',
                            'apache/commons-lang',
                            'zalando/jackson-datatype-money',
                            'zalando/problem/problem-gson',
                            'apache/commons-text',
                            'mybatis/mybatis-dynamic-sql',
                            'mybatis/mybatis-3',
                            'zalando/problem-spring-web/problem-spring-web-autoconfigure'],
                            'valid': ['assertj/assertj-core', 'apache/commons-io'],
                            'test': ['apache/commons-collections',
                            'jqno/equalsverifier',
                            'zalando/problem/jackson-datatype-problem']}
        elif "xcorpus" in dataset_name:
            dataset_split = {'train': ['weka-3-7-9',
                            'guava-21.0',
                            'commons-collections-3.2.1',
                            'ApacheJMeter_core-3.1',
                            'htmlunit-2.8',
                            'velocity-1.6.4',
                            'quartz-1.8.3',
                            'log4j-1.2.16',
                            'informa-0.7.0-alpha2',
                            'javacc-5.0',
                            'openjms-0.7.7-beta-1'],
                            'valid': ['oscache-2.4.1'],
                            'test': ['fitjava-1.1', 'nekohtml-1.9.14', 'trove-2.1.0', 'asm-5.2']}
    # Save dataset
    # with open(join(output_folder, 'dataset_htc_split_v7.json'), 'r') as f:
    #     dataset_split = json.load(f)

    high_tc_data_df_sampled[high_tc_data_df_sampled['program_name'].isin(dataset_split['train'])].to_csv(join(output_folder,
                                                                                            f'{dataset_name}_w_src_train.csv'),
                                                                                    index=False)

    high_tc_data_df_sampled[high_tc_data_df_sampled['program_name'].isin(dataset_split['valid'])].to_csv(join(output_folder,
                                                                                            f'{dataset_name}_w_src_valid.csv'),
                                                                                    index=False)

    high_tc_data_df_sampled[high_tc_data_df_sampled['program_name'].isin(dataset_split['test'])].to_csv(join(output_folder,
                                                                                            f'{dataset_name}_w_src_test.csv'),
                                                                                    index=False)

class FeatureTypes(Enum):
    STRUCT_FEAT = "w_sturct"
    SIG_FEAT = "w_sig"

def read_njr1_programs():
    TRAIN_PROGRAMS = join(PROJECT_DATA_PATH, "njr1/train_programs.txt")
    VALID_PROGRAMS = join(PROJECT_DATA_PATH, "njr1/valid_programs.txt")
    TEST_PROGRAMS = join(PROJECT_DATA_PATH, "njr1/test_programs.txt")

    # Read content from files
    def read_file(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    train_programs = read_file(TRAIN_PROGRAMS)
    valid_programs = read_file(VALID_PROGRAMS)
    test_programs = read_file(TEST_PROGRAMS)

    return train_programs, valid_programs, test_programs

def load_njr1_data(programs: list, cgs_path: str, wiretap_path: str):

    static_cgs = {}
    dynamic_cgs = {}
    for p in programs:
        # p_cg_file = join(cgs_path, p + ".csv")
        p_cg_file = join(wiretap_path, p, "wala_1cfa.csv")
        # if exists(p_cg_file):
            # p_df = pd.read_csv(p_cg_file)
        p_cg_df = pd.read_csv(p_cg_file)
        p_cg_df = p_cg_df.rename(columns={'method': 'source'})
        p_cg_wiretap_df = p_cg_df[p_cg_df['wiretap'] == 1][['source', 'target']]
        p_cg_df = p_cg_df[['source', 'offset', 'target']]
        p_cg_df['source_ln'] = "-1 -1"
        p_cg_df['target_ln'] = "-1 -1"
        
        static_cgs[p] = p_cg_df
        dynamic_cgs[p] = p_cg_wiretap_df
    return static_cgs, dynamic_cgs

def recover_source_code_njr1(output_folder, dataset_name: str, concat_df: pd.DataFrame):
    njr1_train_df = pd.read_csv(join(dirname(output_folder), "train_src2trg_org.csv"))
    njr1_valid_df = pd.read_csv(join(dirname(output_folder), "valid_src2trg_org.csv"))
    njr1_test_df = pd.read_csv(join(dirname(output_folder), "test_src2trg_org.csv"))

    njr1_df = pd.concat([njr1_train_df, njr1_valid_df, njr1_test_df], ignore_index=True)
    
    njr1_methods = njr1_df.drop_duplicates(subset='method').set_index('method')['method_src'].to_dict()
    njr1_targets = njr1_df.drop_duplicates(subset='target').set_index('target')['target_src'].to_dict()
    m2t_src_code = {**njr1_methods, **njr1_targets}
    
    report_training_samples_types(report_nodes_w_src(concat_df))
    for i, r in tqdm(concat_df.iterrows(), total=len(concat_df), desc="Adding source code"):
        if r['method'] != "<boot>":
            if r['method'] in m2t_src_code:
                concat_df.at[i, 'method_src'] = m2t_src_code[r['method']]
        if r['target'] in m2t_src_code:
            concat_df.at[i, 'target_src'] = m2t_src_code[r['target']]
    report_training_samples_types(report_nodes_w_src(concat_df))
    concat_df.to_csv(join(output_folder, f"{dataset_name}_sampled_w_src.csv"), index=False)

if __name__ == "__main__":
    
    dataset_name ='ycorpus'
    FEATURE_SET = None
    if dataset_name == 'ycorpus':
        DATASET_FOLDER = join(PROJECT_DATA_PATH, "/ycorpus/gh_projects_processed_w_deps_v7-5/")
        OUTPUT_FOLDER = join(PROJECT_DATA_PATH, "/ycorpus")
        SRC_FOLDER = join(PROJECT_DATA_PATH, "ycorpus/ghp_sources", "projects_sources_methods")
        JARS_FOLDER = join(PROJECT_DATA_PATH, "ycorpus/gh_projects_processed_jars/")
        DATASET_VERSION = "v7-5"
        ycorpus_dataset_name = f"high_tc_dataset_{FEATURE_SET.value}" if FEATURE_SET is not None else "high_tc_dataset"

        project_static_cg, project_dynamic_cg, project_cg_jars = load_ycorpus_data(DATASET_FOLDER, JARS_FOLDER, limit_projects=2, use_1cfa=True)
        static_dyn_common, _ = find_common_projects(project_static_cg, project_dynamic_cg)
        projects_common_pruned = get_no_static_dyn_edges(set(project_static_cg.keys()).intersection(set(project_dynamic_cg.keys())), 
                                                 project_static_cg, project_dynamic_cg)
        static_dyn_common_pruned_df = create_dataset(OUTPUT_FOLDER, DATASET_VERSION, project_static_cg, project_dynamic_cg)
        concat_dataset = concat_datasets(OUTPUT_FOLDER, f"{ycorpus_dataset_name}_{DATASET_VERSION}", static_dyn_common_pruned_df)
        if FEATURE_SET is not FeatureTypes.SIG_FEAT:
            concat_dataset = recover_dataset_source(OUTPUT_FOLDER, SRC_FOLDER, f"{ycorpus_dataset_name}_{DATASET_VERSION}")
        if FEATURE_SET == FeatureTypes.STRUCT_FEAT:
            gen_struct_features(static_dyn_common, OUTPUT_FOLDER, DATASET_VERSION)
            concat_dataset = add_struct_features(static_dyn_common, concat_dataset, OUTPUT_FOLDER, DATASET_VERSION)
    
        save_dataset_w_src(concat_dataset, OUTPUT_FOLDER, f"{ycorpus_dataset_name}_{DATASET_VERSION}")

    elif dataset_name == 'xcorpus':
        DATASET_FOLDER = join(PROJECT_DATA_PATH, "xcorpus/wiretap_xcorpus_v4/")
        OUTPUT_FOLDER = join(PROJECT_DATA_PATH, "xcorpus")
        SRC_FOLDER = join(PROJECT_DATA_PATH, "xcorpus/xcorpus_sources/xcorpus-src")
        SECOND_SRC_FOLDER = None
        DATASET_VERSION = "v4-5"
        xcorpus_dataset_name = f"xcorpus_dataset_{FEATURE_SET.value}" if FEATURE_SET is not None else "xcorpus_dataset" 

        project_static_cg, project_dynamic_cg = load_xcorpus_data(DATASET_FOLDER, use_1cfa=True)
        static_dyn_common, _ = find_common_projects(project_static_cg, project_dynamic_cg)
        static_dyn_common_pruned_df = create_dataset(OUTPUT_FOLDER, DATASET_VERSION, project_static_cg, project_dynamic_cg)
        concat_dataset = concat_datasets(OUTPUT_FOLDER, f"{xcorpus_dataset_name}_{DATASET_VERSION}", static_dyn_common_pruned_df)
        if FEATURE_SET is not FeatureTypes.SIG_FEAT:
            concat_dataset = recover_dataset_source(OUTPUT_FOLDER, SRC_FOLDER, f"{xcorpus_dataset_name}_{DATASET_VERSION}",
                                                    SECOND_SRC_FOLDER)
        if FEATURE_SET == FeatureTypes.STRUCT_FEAT:
            gen_struct_features(static_dyn_common, OUTPUT_FOLDER, DATASET_VERSION)
            concat_dataset = add_struct_features(static_dyn_common, concat_dataset, OUTPUT_FOLDER, DATASET_VERSION)
       
        save_dataset_w_src(concat_dataset ,OUTPUT_FOLDER, f"{xcorpus_dataset_name}_{DATASET_VERSION}")

    elif dataset_name == "njr1":
        NJR1_FOLDER = join(PROJECT_DATA_PATH, "njr1/raw_data")
        ONE_CFA_CGS = join(PROJECT_DATA_PATH, "njr1/cgs/1cfa")
        DATASET_VERSION = "v1.0"
        OUTPUT_FOLDER =  join(PROJECT_DATA_PATH, "njr1/dataset")
        njr1_dataset_name = "njr1_dataset"

        train_njr1_programs, valid_njr1_programs, test_njr1_programs = read_njr1_programs()

        project_static_cg, project_dynamic_cg = load_njr1_data(train_njr1_programs+valid_njr1_programs+test_njr1_programs,
                                                                ONE_CFA_CGS, NJR1_FOLDER)
        static_dyn_common, _ = find_common_projects(project_static_cg, project_dynamic_cg)
        static_dyn_common_pruned_df = create_dataset(OUTPUT_FOLDER, DATASET_VERSION, project_static_cg, project_dynamic_cg)
        concat_dataset = concat_datasets(OUTPUT_FOLDER, f"{njr1_dataset_name}_{DATASET_VERSION}", static_dyn_common_pruned_df)
        recover_source_code_njr1(OUTPUT_FOLDER, f"{njr1_dataset_name}_{DATASET_VERSION}", concat_dataset)
        save_dataset_w_src(concat_dataset, OUTPUT_FOLDER, f"{njr1_dataset_name}_{DATASET_VERSION}", dataset_split={'train': train_njr1_programs,
                                                                                                                   'valid': valid_njr1_programs,
                                                                                                                    'test': test_njr1_programs})
