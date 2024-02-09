from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import statistics
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def compute_recall_precision_f1(df: pd.DataFrame):
    """
    Computes recall, precison, and f1-score for static Wala's call graphs
    """
    try:
        r = df[(df['wala-cge-0cfa-noreflect-intf-trans'] == 1) & (df['wiretap'] == 1)].shape[0] / df[df['wiretap'] == 1].shape[0]
    except ZeroDivisionError:
        r = 0

    try:
        p = df[(df['wala-cge-0cfa-noreflect-intf-trans'] == 1) & (df['wiretap'] == 1)].shape[0] / df[df['wala-cge-0cfa-noreflect-intf-trans'] == 1].shape[0]
    except ZeroDivisionError:
        p = 0

    try:
        f1 = (2 * p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0

    return p, r, f1

def compute_recall_precision_f1_f2(df: pd.DataFrame):
    """
    Computes recall, precison, f1-score and f2-score for static Wala's call graphs
    """
    try:
        r = df[(df['wala-cge-0cfa-noreflect-intf-trans'] == 1) & (df['wiretap'] == 1)].shape[0] / df[df['wiretap'] == 1].shape[0]
    except ZeroDivisionError:
        r = 0

    try:
        p = df[(df['wala-cge-0cfa-noreflect-intf-trans'] == 1) & (df['wiretap'] == 1)].shape[0] / df[df['wala-cge-0cfa-noreflect-intf-trans'] == 1].shape[0]
    except ZeroDivisionError:
        p = 0

    f1 = compute_f1_score(r, p)

    f2 = compute_f2_score(r, p)

    return p, r, f1, f2

def compute_f1_score(r, p):
    try:
        f1 = (2 * p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0
    return f1

def compute_f2_score(r: float, p: float) -> float:
    try:
        f2 = ((1+2**2) * p * r) / (2**2 * p + r)
    except ZeroDivisionError:
        f2 = 0
    return f2

def compute_model_out_recall_precision_f1(df: pd.DataFrame, model_out_col='m_out'):
    try:
        r = df[(df[model_out_col] == 1) & (df['wiretap'] == 1)].shape[0] / df[df['wiretap'] == 1].shape[0]
    except ZeroDivisionError:
        r = 0
    try:
        p = df[(df[model_out_col] == 1) & (df['wiretap'] == 1)].shape[0] / df[df[model_out_col] == 1].shape[0]
    except ZeroDivisionError:
        p = 0

    try:
        f1 = (2*p*r) / (p+r) if (p+r) != 0 else 0
    except ZeroDivisionError:
        f1 = 0

    return p, r, f1


def compute_prune_add_agree_metrics(program_df: pd.DataFrame, model_out_col='m_out'):
    prune = accuracy_score(program_df[(program_df['wiretap'] == 0) & (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 1)]['wiretap'],
                                program_df[(program_df['wiretap'] == 0) & (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 1)][model_out_col])
    add = accuracy_score(program_df[(program_df['wiretap'] == 1) & (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 0)]['wiretap'],
                                program_df[(program_df['wiretap'] == 1) & (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 0)][model_out_col])
    agree = accuracy_score(program_df[(program_df['wiretap'] == 1) & (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 1)]['wiretap'],
                                program_df[(program_df['wiretap'] == 1) & (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 1)][model_out_col])

    return prune, add, agree


def compute_eval_metrics(test_programs_list: list, df_test: pd.DataFrame, model_out_col='m_out') -> pd.DataFrame:
    """
    Computes the evaluation metrics for Both vanilla Wala and the model's CGs
    """

    df_test = df_test.drop_duplicates()
    test_programs_results = []
    test_programs_list = df_test['program_name'].unique()
    for i, p in tqdm(enumerate(test_programs_list), leave=False, total=len(test_programs_list)):
        # try:
        program_df = df_test[df_test['program_name'] == p]
        program_df.reset_index(inplace=True)
        w_prec, w_rec, w_f1 = compute_recall_precision_f1(program_df)
        prec, rec, f1 = compute_model_out_recall_precision_f1(program_df, model_out_col=model_out_col)
        prune, add, agree = compute_prune_add_agree_metrics(program_df, model_out_col=model_out_col)
        # Skip programs with no edges
        if w_prec + w_rec + w_f1 + prec + rec + f1 != 0.0:
            test_programs_results.append([p, w_prec, w_rec, w_f1, prec, rec, f1, prune, add, agree])
        # except ZeroDivisionError:
        #     print("Could not evaluate the program " + p)
    
    test_programs_results_df = pd.DataFrame(test_programs_results, columns=['program', 'wala_prec', 'wala_rec', 'wala_f1',
                                                                        'precision', 'recall', 'f1', 'prune', 'add', 'agree'])
    test_programs_results_df.fillna(0, inplace=True)

    return test_programs_results_df


def compute_eval_metrics_wala(df_test: pd.DataFrame):
    """
    Computes the evaluation metrics for WALA's CGs
    """
    w_prec_l, w_rec_l, w_f1_l = [], [], []
    test_programs_list = df_test['program_name'].unique()
    for i, p in enumerate(test_programs_list):
        # try:
        program_df = df_test[df_test['program_name'] == p]
        program_df = program_df[program_df['method'] != "<boot>"]
        program_df = program_df[program_df['target'] != 'com/ibm/wala/FakeRootClass.fakeWorldClinit:()V']
        program_df.reset_index(inplace=True)
        w_prec, w_rec, w_f1 = compute_recall_precision_f1(program_df)
        print(f"Program {p} | Wala prec: {w_prec:.2f} Wala rec: {w_rec:.2f} Wala f1: {w_f1:.2f}")
        w_prec_l.append(w_prec)
        w_rec_l.append(w_rec)
        w_f1_l.append(w_f1)
        # except ZeroDivisionError:
        #     print("Could not evaluate the program " + p)
    print(f"Wala prec: {statistics.mean(w_prec_l):.2f} Wala rec: {statistics.mean(w_rec_l):.2f} Wala f1: {statistics.mean(w_f1_l):.2f}")


def compute_eval_metrics_paper(test_programs_list: list, df_test: pd.DataFrame, model_out_col='m_out', prune_prob: float=None) -> pd.DataFrame:
    """
    Computes the evaluation metrics for Both vanilla Wala and the model's CGs
    """

    df_test = df_test.drop_duplicates()
    test_programs_results = []
    test_programs_list = df_test['program_name'].unique()
    for i, p in tqdm(enumerate(test_programs_list), leave=False, total=len(test_programs_list)):
        # try:
        program_df = df_test[df_test['program_name'] == p]
        program_df.reset_index(inplace=True)
        # program_df[(program_df['wiretap'] == 1) & \
        #      (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 0) & \
        #          (program_df[model_out_col] == 1)][model_out_col] = 0
        added_edges = (program_df['wiretap'] == 1) & \
            (program_df['wala-cge-0cfa-noreflect-intf-trans'] == 0) & \
            (program_df[model_out_col] == 1)
        program_df.loc[added_edges, model_out_col] = 0

        if prune_prob is not None:
            program_df['m_out_p_prune'] = program_df['m_out_p'].apply(lambda x : 1 - x)
            program_df['m_out'] = program_df['m_out_p_prune'].apply(lambda x: 0 if x > prune_prob else 1)

        w_prec, w_rec, w_f1 = compute_recall_precision_f1(program_df)
        prec, rec, f1 = compute_model_out_recall_precision_f1(program_df, model_out_col=model_out_col)
        prune, add, agree = compute_prune_add_agree_metrics(program_df, model_out_col=model_out_col)
        # Skip programs with no edges
        if w_prec + w_rec + w_f1 + prec + rec + f1 != 0.0:
            test_programs_results.append([p, w_prec, w_rec, w_f1, prec, rec, f1, prune, add, agree])
        # except ZeroDivisionError:
        #     print("Could not evaluate the program " + p)
    
    test_programs_results_df = pd.DataFrame(test_programs_results, columns=['program', 'wala_prec', 'wala_rec', 'wala_f1',
                                                                        'precision', 'recall', 'f1', 'prune', 'add', 'agree'])
    test_programs_results_df.fillna(0, inplace=True)

    return test_programs_results_df


def report_eval_metrics(test_programs_results_df: pd.DataFrame):
    prec, recall, f1 = test_programs_results_df['precision'].mean(), test_programs_results_df['recall'].mean(), \
    test_programs_results_df['f1'].mean()
    wala_prec, wala_rec, wala_f1 = test_programs_results_df['wala_prec'].mean(), \
    test_programs_results_df['wala_rec'].mean(), test_programs_results_df['wala_f1'].mean()
    prune, add, agree = test_programs_results_df['prune'].mean(), \
    test_programs_results_df[test_programs_results_df['wala_rec']< 1.0]['add'].mean(), test_programs_results_df['agree'].mean()
    print(f"Precision: {prec:.2f} Recall: {recall:.2f} F1: {f1:.2f}")
    print(f"Wala prec: {wala_prec:.2f} Wala rec: {wala_rec:.2f} Wala f1: {wala_f1:.2f}")
    print(f"Prune: {prune:.2f} Add: {add:.2f} agree: {agree:.2f}")
    return prec, recall, f1, wala_prec, wala_rec, wala_f1

def report_eval_metrics_paper(test_programs_results_df: pd.DataFrame):
    prec, recall = test_programs_results_df['precision'].mean(), test_programs_results_df['recall'].mean()
    f1 = compute_f1_score(recall, prec)
    f2 = compute_f2_score(recall, prec)
    wala_prec, wala_rec = test_programs_results_df['wala_prec'].mean(), test_programs_results_df['wala_rec'].mean()
    wala_f1 = compute_f1_score(wala_rec, wala_prec)
    wala_f2 = compute_f2_score(wala_rec, wala_prec)
    # prune, add, agree = test_programs_results_df['prune'].mean(), \
    # test_programs_results_df[test_programs_results_df['wala_rec']< 1.0]['add'].mean(), test_programs_results_df['agree'].mean()
    print(f"Precision: {prec:.2f} Recall: {recall:.2f} F1: {f1:.2f} F2: {f2:.2f}")
    print(f"Wala prec: {wala_prec:.2f} Wala rec: {wala_rec:.2f} Wala f1: {wala_f1:.2f} Wala F2: {wala_f2:.2f}")
    # print(f"Prune: {prune:.2f} Add: {add:.2f} agree: {agree:.2f}")
    return prec, recall, f1, f2, wala_prec, wala_rec, wala_f1, wala_f2
