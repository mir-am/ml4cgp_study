from os.path import join
from src.eval import report_eval_metrics_paper, compute_eval_metrics_paper
from src.utils import read_json
from src import PROJECT_DATA_PATH
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statistics
import argparse

RESULTS_FOLDER = join(PROJECT_DATA_PATH, "results_new/")

def compute_average(rows):
    def extract_value(val):
        match = re.search(r'\\textbf{([\d.]+)}(\\\\)?', val)
        if match:
            return float(match.group(1))
        elif val == '-':
            return 0.0
        else:
            try:
                # Try to match just the number possibly followed by "\\"
                match_num = re.search(r'([\d.]+)(\\\\)?', val)
                if match_num:
                    return float(match_num.group(1))
                return float(val)
            except ValueError:
                return 0

    numeric_values = []
    for row in rows:
        components = row.split("&")[1:]
        numeric_row = [extract_value(val) for val in components]
        numeric_values.append(numeric_row)

    averages = []
    for col in zip(*numeric_values):
        col_average = sum(col) / len(col)
        
        if all(val == 0.0 for val in col):
            averages.append('-')
        else:
            averages.append("{:.2f}".format(col_average))

    average_str = "\\textbf{Average} & " + " & ".join(averages) + " \\\\"

    return average_str

def bold_max_values_in_latex_table(rows):
    # Convert the string values to float and ignore non-numeric values (like model names)
    def to_float(val):
        try:
            return float(val.strip())
        except ValueError:
            return None

    numeric_values = [[to_float(cell) for cell in row.split('&')[1:]] for row in rows]
    max_values = [max(filter(None, col)) for col in zip(*numeric_values)]

    bolded_rows = []
    for row in rows:
        components = row.split('&')
        model_name = components[0]
        values = components[1:]
        
        bolded_values = []
        for i, val in enumerate(values):
            val = val.strip()
            if to_float(val) == max_values[i]:
                bolded_values.append("\\textbf{" + val + "}")
            else:
                bolded_values.append(val)
        
        bolded_row = model_name + " & " + " & ".join(bolded_values) + " \\\\"
        bolded_rows.append(bolded_row)

    return bolded_rows

def gen_RQ1_results():
    models = {
    "RC": "\\textbf{Random Classifier}", 
    "RF": "\\textbf{CGPruner}", 
    "codebert_ws": "\\textbf{AutoPruner}", 
    "codebert": "\\textbf{CodeBERT}", 
    "codet5": "\\textbf{CodeT5}", 
    "codet5_plus": "\\textbf{CodeT5+}"
    }
    
    models_latex_r = []
    w_l = "\\textbf{Wala}"
    for m in ["RC", "RF", "codebert_ws", "codebert", "codet5", "codet5_plus"]:
        model_l = models[m]
        for d in ['org', 'xcorpus', 'dhtc_w_src', 'nyx_w_src']:
            if 'xcorpus' in d:
                if m != "RF":
                    d = "now_" + d
                else:
                    d = d + "_w_struct"
            elif 'dhtc' in d or 'nyx' in d:
                if m == "RF" and 'nyx' in d:
                    d = "now_" + d.replace("_w_src", "") + "_w_struct"
                elif m == "RF":
                    d = d.replace("_w_src", "") + "_w_struct"
                else:
                    if "ws" not in m:
                        d = "now_" + d
                    else:
                        d = "now_" + d.replace("_w_src", "")
            elif 'org' in d and m == "RC":
                d = "now_" + d
            if "RF" in m:
                res_f = join(RESULTS_FOLDER, f"test_src2trg_w_preds_RF_CG_pruner_best_{d}.csv")
            else:
                # res_f = join(RESULTS_FOLDER, f"test_programs_metrics_{m}_pruner_{d}{'_w_struct'if 'ws' in m else ''}.csv")
                res_f = join(RESULTS_FOLDER, f"test_src2trg_w_preds_{m}_pruner_{d}{'_w_struct'if 'ws' in m else ''}.csv")
            try:
                df_res = pd.read_csv(res_f)
                df_res = compute_eval_metrics_paper([], df_res)
                p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(df_res)
                p, r, f1, f2, w_p, w_r, w_f1, w_f2 = ["{:.2f}".format(val) for val in (p, r, f1, f2, w_p, w_r, w_f1, w_f2)]
            except FileNotFoundError as e:
                p, r, f1, f2, w_p, w_r, w_f1, w_f2 = "-", "-", "-", "-", "-", "-", "-", "-"

            model_l = model_l + " & " + p + " & " + r + " & " + f1 + " & " + f2
            if w_l.count("&") != 16 and m != "RF": 
                w_l = w_l + " & " + w_p + " & " + w_r + " & " + w_f1 + " & " + w_f2

            # print(model_l)
        models_latex_r.append(model_l)
    models_latex_r = bold_max_values_in_latex_table(models_latex_r)
    for l in models_latex_r:
        print(l)
    print("\\midrule")
    print(compute_average(models_latex_r))
    print("\\midrule")
    print(w_l + " \\\\")

def gen_RQ2_results():
    for m in ['codebert', 'codet5']:
        output_fig_path = join(RESULTS_FOLDER, f"{m}_rq2_fig.pdf")
        rq2_data = {
            'Group': ['0.0', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', 'Wala (0-CFA)'],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'F2': []
        }
        for w in ['now', 'C06', 'C07', 'C08', 'C09', 'C095', 'C099', 'Wala (0-CFA)']:
            if w != "Wala (0-CFA)":
                df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{m}_pruner_{w}_nyx_w_src.csv"))
                df_res = compute_eval_metrics_paper([], df_res)
                p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(df_res)
                
                rq2_data["Precision"].append(round(p, 2))
                rq2_data["Recall"].append(round(r, 2))
                rq2_data["F1"].append(round(f1, 2))
                rq2_data["F2"].append(round(f2, 2))
                
            else:
                res_f = pd.read_csv(join(RESULTS_FOLDER, "test_programs_metrics_codet5_plus_pruner_now_nyx_w_src.csv"))
                p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(res_f)
                rq2_data["Precision"].append(round(w_p, 2))
                rq2_data["Recall"].append(round(w_r, 2))
                rq2_data["F1"].append(round(w_f1, 2))
                rq2_data["F2"].append(round(w_f2, 2))
    
        df = pd.DataFrame(rq2_data)
        df_melt = df.melt(id_vars='Group', var_name='Metrics', value_name='Value')

        # Create bar plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Group', y='Value', hue='Metrics', data=df_melt)
        line_x = (df['Group'].index[-2] + df['Group'].index[-1]) / 2
        ax.axvline(line_x, color='black', linestyle='--') 
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

        plt.ylim(0, 1)
        plt.xlabel("Weight")
        plt.ylabel("Score")
        plt.savefig(output_fig_path, dpi=256)
        print(f"Saved the figure for {m} with different weights at {output_fig_path}")
        # Create line plot
        # plt.figure(figsize=(10, 6))
        # ax = sns.lineplot(x='Group', y='F2', data=df, marker="o")
        # line_x = (df['Group'].index[-2] + df['Group'].index[-1]) / 2
        # ax.axvline(line_x, color='black', linestyle='--') 
        # # Annotate each point on the line with its Y value
        # for x, y in zip(df['Group'], df['F2']):
        #     ax.text(x, y + 0.02, f"{y:.2f}", ha='center')  # The "+ 0.02" offsets the text upwards for better visibility

        # plt.ylim(0, 1)
        # plt.savefig(join(RESULTS_FOLDER, f"{m}_rq2_fig_f2.pdf", dpi=256))

def gen_RQ2_1_results():
    for m in ['codebert', 'codet5']:
        output_fig_path = join(RESULTS_FOLDER, f"{m}_rq2_1_fig_f2.pdf")
        rq2_data = {
                'Group': ['baseline', '0.6', '0.7', '0.8', '0.9', '0.95', 'Wala (0-CFA)'], #, 'CT5+ (C08)'],
                'Precision': [],
                'Recall': [],
                'F1': [],
                'F2': []
            }
        for w in ['now', 'Wala (0-CFA)']: #, 'CT5+ (C08)']:
            if w != "Wala (0-CFA)":
                for prune_p in [None, 0.6, 0.7, 0.8, 0.9, 0.95]:
                    df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{m}_pruner_{w}_nyx_w_src.csv"))
                    df_res = compute_eval_metrics_paper([], df_res, prune_prob=prune_p)
                    p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(df_res)
                    
                    rq2_data["Precision"].append(round(p, 2))
                    rq2_data["Recall"].append(round(r, 2))
                    rq2_data["F1"].append(round(f1, 2))
                    rq2_data["F2"].append(round(f2, 2))
                
            else:
                res_f = pd.read_csv(join(RESULTS_FOLDER, "test_programs_metrics_codet5_plus_pruner_now_nyx_w_src.csv"))
                p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(res_f)
                rq2_data["Precision"].append(round(w_p, 2))
                rq2_data["Recall"].append(round(w_r, 2))
                rq2_data["F1"].append(round(w_f1, 2))
                rq2_data["F2"].append(round(w_f2, 2))
        
        df = pd.DataFrame(rq2_data)
        df_melt = df.melt(id_vars='Group', var_name='Metrics', value_name='Value')

        # Create bar plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Group', y='Value', hue='Metrics', data=df_melt)
        line_x = (df['Group'].index[-2] + df['Group'].index[-1]) / 2
        ax.axvline(line_x, color='black', linestyle='--') 
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

        plt.ylim(0, 1)
        plt.xlabel("Confidence levels")
        plt.ylabel("Score")
        plt.savefig(output_fig_path, dpi=256)
        print(f"Saved the figure for {m} with different conf. levels at {output_fig_path}")
        # plt.savefig(f"src/scripts/tmp/{m}_rq2_1_fig.pdf", dpi=256)

def gen_RQ3_results():
    models = {
    "codebert": "\\textbf{CodeBERT}", 
    "codet5": "\\textbf{CodeT5}", 
    }
    print("\midrule\n\multicolumn{5}{c}{0-CFA} \\\ \n\midrule")
    # 0-CFA
    models_latex_r_0cfa = []
    w_l_0cfa = "\\textbf{Wala}"
    for m in ['codebert', 'codet5']:
        model_l = models[m]
        df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{m}_pruner_now_nyx_w_src.csv"))
        df_res = compute_eval_metrics_paper([], df_res, prune_prob=0.95)
        p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(df_res)
        p, r, f1, f2, w_p, w_r, w_f1, w_f2 = ["{:.2f}".format(val) for val in (p, r, f1, f2, w_p, w_r, w_f1, w_f2)]
        model_l = model_l + " & " + p + " & " + r + " & " + f1 + " & " + f2
        models_latex_r_0cfa.append(model_l)
        if w_l_0cfa.count("&") != 4: 
                w_l_0cfa = w_l_0cfa + " & " + w_p + " & " + w_r + " & " + w_f1 + " & " + w_f2
    models_latex_r_0cfa = bold_max_values_in_latex_table(models_latex_r_0cfa)
    for r in models_latex_r_0cfa:
        print(r)
    print(w_l_0cfa + " \\\\")
    print("\midrule\n\multicolumn{5}{c}{1-CFA} \\\ \n\midrule")
    # 1-CFA
    models_latex_r_1cfa = []
    w_l_1cfa = "\\textbf{Wala}"
    for m in ['codebert', 'codet5']:
        model_l = models[m]
        df_res = pd.read_csv(join(RESULTS_FOLDER, f"test_src2trg_w_preds_{m}_pruner_now_nyx_1cfa_w_src.csv"))
        df_res = compute_eval_metrics_paper([], df_res, prune_prob=0.95)
        p, r, f1, f2, w_p, w_r, w_f1, w_f2 = report_eval_metrics_paper(df_res)
        p, r, f1, f2, w_p, w_r, w_f1, w_f2 = ["{:.2f}".format(val) for val in (p, r, f1, f2, w_p, w_r, w_f1, w_f2)]
        model_l = model_l + " & " + p + " & " + r + " & " + f1 + " & " + f2
        models_latex_r_1cfa.append(model_l)
        if w_l_1cfa.count("&") != 4: 
                w_l_1cfa = w_l_1cfa + " & " + w_p + " & " + w_r + " & " + w_f1 + " & " + w_f2
    models_latex_r_1cfa = bold_max_values_in_latex_table(models_latex_r_1cfa)
    for r in models_latex_r_1cfa:
        print(r)
    print(w_l_1cfa + " \\\\")

def gen_RQ3_1_results():
    runtime_cg_log_file = "/mnt/data/amir_projects/ml4cg/src/scripts/logs/gen_cg_0_1_cfa_run_time"
    zero_cfa_feat_ext_file = join(PROJECT_DATA_PATH, "results/feat_ext_t_nyx_w_src_0cfa.json")
    zero_cfa_models_inf_file = join(PROJECT_DATA_PATH, "results/models_inf_t_nyx_w_src_0cfa.json")
    one_cfa_feat_ext_file = join(PROJECT_DATA_PATH, "results/feat_ext_t_nyx_w_src_1cfa.json")
    one_cfa_models_inf_file = join(PROJECT_DATA_PATH, "results/models_inf_t_nyx_w_src_1cfa.json")

    data = {}
    data['zero_cfa_feat_ext'] = read_json(zero_cfa_feat_ext_file)
    data['zero_cfa_models_inf'] = read_json(zero_cfa_models_inf_file)
    data['one_cfa_feat_ext'] = read_json(one_cfa_feat_ext_file)
    data['one_cfa_models_inf'] = read_json(one_cfa_models_inf_file)
    test_programs = list(data['zero_cfa_models_inf']['codebert'].keys())

    zerocfa_regex = r"ZeroCFA - Generated a call graph .+ in ([\d.]+) seconds"
    onecfa_regex = r"OneCFA - Generated a call graph .+ in ([\d.]+) seconds"
    program_name_regex = r"Building a call graph for (.+)"
    zerocfa_times = []
    onecfa_times = []
    program_names = []
    IGNORE_TEST_PROGRAM = True

    with open(runtime_cg_log_file, "r") as f:
        p_name = ""
        for line in f:
            # Check for program name
            program_name_match = re.search(program_name_regex, line)
            if program_name_match:
                p_name = program_name_match.group(1)
                if "xcorpus-extension-20170313" in p_name or 'qualitas_corpus_20130901' in p_name:
                    p_name = p_name.split("/")[1]
                program_names.append(p_name)
            if IGNORE_TEST_PROGRAM or p_name in test_programs:
                zerocfa_match = re.search(zerocfa_regex, line)
                if zerocfa_match:
                    zerocfa_times.append(float(zerocfa_match.group(1)))

                # Check for OneCFA time
                onecfa_match = re.search(onecfa_regex, line)
                if onecfa_match:
                    onecfa_times.append(float(onecfa_match.group(1)))

    zero_cfa_t_avg = f"{statistics.mean(zerocfa_times):.1f}"
    zero_cfa_t_std = f"{statistics.stdev(zerocfa_times):.1f}"
    one_cfa_t_avg = f"{statistics.mean(onecfa_times):.1f}"
    one_cfa_t_std = f"{statistics.stdev(onecfa_times):.1f}"

    # print(f"ZeroCFA avg. time: {statistics.mean(zerocfa_times):.2f} +- {statistics.stdev(zerocfa_times):.2f} sec.")
    # print(f"OneCFA avg. time: {statistics.mean(onecfa_times):.2f} +- {statistics.stdev(onecfa_times):.2f} sec.")
    
    print("\midrule\n\multicolumn{5}{@{}l}{\it 0-CFA} \\\\ \n\midrule")
    # cb_total_t_sum_0cfa = zerocfa_times + list(data['zero_cfa_feat_ext'].values()) + list(data['zero_cfa_models_inf']['codebert'].values())
    cb_total_t_sum_0cfa = [sum(v) for v in zip(zerocfa_times, list(data['zero_cfa_feat_ext'].values()), list(data['zero_cfa_models_inf']['codebert'].values()))]
    # ct5_total_t_sum_0cfa = zerocfa_times + list(data['zero_cfa_feat_ext'].values()) + list(data['zero_cfa_models_inf']['codet5'].values())
    ct5_total_t_sum_0cfa = [sum(v) for v in zip(zerocfa_times, list(data['zero_cfa_feat_ext'].values()), list(data['zero_cfa_models_inf']['codet5'].values()))]
    print("\\textbf{CodeBERT}" + " & " + f"{zero_cfa_t_avg} $\pm$ {zero_cfa_t_std}" +  " & " + \
           f"{statistics.mean(data['zero_cfa_feat_ext'].values()):.1f} $\pm$ {statistics.stdev(data['zero_cfa_feat_ext'].values()):.1f}" + \
                " & "  +  f"{statistics.mean(data['zero_cfa_models_inf']['codebert'].values()):.1f} $\pm$ {statistics.stdev(data['zero_cfa_models_inf']['codebert'].values()):.1f}" + " & " + \
                      f"{statistics.mean(cb_total_t_sum_0cfa):.1f} $\pm$ {statistics.stdev(cb_total_t_sum_0cfa):.1f}"+ "\\\\")
    print("\\textbf{CodeT5}" + " & " + f"{zero_cfa_t_avg} $\pm$ {zero_cfa_t_std}" +  " & " + \
           f"{statistics.mean(data['zero_cfa_feat_ext'].values()):.1f} $\pm$ {statistics.stdev(data['zero_cfa_feat_ext'].values()):.1f}" + \
                " & "  +  f"{statistics.mean(data['zero_cfa_models_inf']['codet5'].values()):.1f} $\pm$ {statistics.stdev(data['zero_cfa_models_inf']['codet5'].values()):.1f}" + " & " + \
                      f"{statistics.mean(ct5_total_t_sum_0cfa):.1f} $\pm$ {statistics.stdev(ct5_total_t_sum_0cfa):.1f}" + " \\\\")
    print("\midrule\n\multicolumn{5}{@{}l}{\it 1-CFA} \\\\ \n\midrule")
    # cb_total_t_sum_1cfa = onecfa_times + list(data['one_cfa_feat_ext'].values()) + list(data['one_cfa_models_inf']['codebert'].values())
    cb_total_t_sum_1cfa = [sum(v) for v in zip(onecfa_times, list(data['one_cfa_feat_ext'].values()), list(data['one_cfa_models_inf']['codebert'].values()))]
    # ct5_total_t_sum_1cfa = onecfa_times + list(data['one_cfa_feat_ext'].values()) + list(data['one_cfa_models_inf']['codet5'].values())
    ct5_total_t_sum_1cfa = [sum(v) for v in zip(onecfa_times, list(data['one_cfa_feat_ext'].values()), list(data['one_cfa_models_inf']['codet5'].values()))]
    print("\\textbf{CodeBERT}" + " & " + f"{one_cfa_t_avg} $\pm$ {one_cfa_t_std}" +  " & " + \
           f"{statistics.mean(data['one_cfa_feat_ext'].values()):.1f} $\pm$ {statistics.stdev(data['one_cfa_feat_ext'].values()):.1f}" + \
                " & "  +  f"{statistics.mean(data['one_cfa_models_inf']['codebert'].values()):.1f} $\pm$ {statistics.stdev(data['one_cfa_models_inf']['codebert'].values()):.1f}" + " & " + \
                      f"{statistics.mean(cb_total_t_sum_1cfa):.1f} $\pm$ {statistics.stdev(cb_total_t_sum_1cfa):.1f}"+ "\\\\")
    print("\\textbf{CodeT5}" + " & " + f"{one_cfa_t_avg} $\pm$ {one_cfa_t_std}" +  " & " + \
           f"{statistics.mean(data['one_cfa_feat_ext'].values()):.1f} $\pm$ {statistics.stdev(data['one_cfa_feat_ext'].values()):.1f}" + \
                " & "  +  f"{statistics.mean(data['one_cfa_models_inf']['codet5'].values()):.1f} $\pm$ {statistics.stdev(data['one_cfa_models_inf']['codet5'].values()):.1f}" + " & " + \
                      f"{statistics.mean(ct5_total_t_sum_1cfa):.1f} $\pm$ {statistics.stdev(ct5_total_t_sum_1cfa):.1f}" + " \\\\")


def gen_RQ4_results():
    vuln_analysis_json_file = join(RESULTS_FOLDER, "nyx_corpus_vuln_analysis.json")
    vuln_analysis_results = read_json(vuln_analysis_json_file)
    def compute_avg(cg_data_name: str):
        avg_reach_vuln_paths =  statistics.mean([p['numOfReachablePaths'] for p in vuln_analysis_results[cg_data_name]])
        avg_reach_vuln_nodes = statistics.mean([p['numOfReachableVulnNodes'] for p in vuln_analysis_results[cg_data_name]])
        avg_analysis_time = statistics.mean([p['reachabilityAnalysisTime'] for p in vuln_analysis_results[cg_data_name]])
        std_analysis_time = statistics.stdev([p['reachabilityAnalysisTime'] for p in vuln_analysis_results[cg_data_name]])
        avg_cg_nodes = statistics.mean([p['numOfCGNodes'] for p in vuln_analysis_results[cg_data_name]])
        avg_cg_edges = statistics.mean([p['numOfCGEdges'] for p in vuln_analysis_results[cg_data_name]])
        return avg_reach_vuln_paths, avg_reach_vuln_nodes, avg_analysis_time, std_analysis_time, avg_cg_nodes, avg_cg_edges

    wala_avg_reach_vuln_paths, wala_avg_reach_vuln_nodes, wala_avg_analysis_time, wala_std_analysis_time, wala_cg_nodes, wala_cg_edges = compute_avg("wala")
    cb_avg_reach_vuln_paths, cb_avg_reach_vuln_nodes, cb_avg_analysis_time, cb_std_analysis_time, cb_cg_nodes, cb_cg_edges = compute_avg("codebert")
    cbw_avg_reach_vuln_paths, cbw_avg_reach_vuln_nodes, cbw_avg_analysis_time, cbw_std_analysis_time, cbw_cg_nodes, cbw_cg_edges = compute_avg("codebert_C99")
    ct5_avg_reach_vuln_paths, ct5_avg_reach_vuln_nodes, ct5_avg_analysis_time, ct5_std_analysis_time, ct5_cg_nodes, ct5_cg_edges = compute_avg("codet5")
    ct5w_avg_reach_vuln_paths, ct5w_avg_reach_vuln_nodes, ct5w_avg_analysis_time, ct5w_std_analysis_time, ct5w_cg_nodes, ct5w_cg_edges = compute_avg("codet5_C99")
    # rc_avg_reach_vuln_paths, rc_avg_reach_vuln_nodes, rc_avg_analysis_time, rc_std_analysis_time, rc_cg_nodes, rc_cg_edges = compute_avg("RC")

    print("\\textbf{Wala} & " + f"{wala_cg_edges:.1f} & {wala_cg_nodes:.1f} " + " & " + f"{wala_avg_reach_vuln_paths:.1f}"  + " & " + \
           f"{wala_avg_reach_vuln_nodes:.1f}/100"  + " & " +  f"{wala_avg_analysis_time:.1f} $\pm$ {wala_std_analysis_time:.1f}" + " \\\\")
    print("\midrule\n\multicolumn{6}{@{}c@{}}{{\it Conservative Pruning} \hfill ($> 0.95$ confidence)} \\\\\n\midrule")
    print("\\textbf{CodeBERT} & " + f"{cb_cg_edges:.1f} & {cb_cg_nodes:.1f}" + " & " + f"{cb_avg_reach_vuln_paths:.1f}"  + " & " + f"{cb_avg_reach_vuln_nodes:.1f}/100"  + \
           " & " +  f"{cb_avg_analysis_time:.1f} $\pm$ {cb_std_analysis_time:.1f}" + " \\\\")
    print("\\textbf{CodeT5} & " + f"{ct5_cg_edges:.1f} & {ct5_cg_nodes:.1f} " + " & " + f"{ct5_avg_reach_vuln_paths:.1f}"  + " & " + f"{ct5_avg_reach_vuln_nodes:.1f}/100"  + \
           " & " +  f"{ct5_avg_analysis_time:.1f} $\pm$ {ct5_std_analysis_time:.1f}" + " \\\\")
    print("\midrule\n\multicolumn{6}{@{}c@{}}{{\it Paranoid Pruning} \hfill ($0.99$ weight, $> 0.95$ confidence)} \\\\\n\midrule")
    
    print("\\textbf{CodeBERT} & " + f"{cbw_cg_edges:.1f} & {cbw_cg_nodes:.1f}" + " & " + f"{cbw_avg_reach_vuln_paths:.1f}"  + " & " + f"{cbw_avg_reach_vuln_nodes:.1f}/100"  + \
           " & " +  f"{cbw_avg_analysis_time:.1f} $\pm$ {cbw_std_analysis_time:.1f}" + " \\\\")
    print("\\textbf{CodeT5} & " + f"{ct5w_cg_edges:.1f} & {ct5w_cg_nodes:.1f}" + " & " + f"{ct5w_avg_reach_vuln_paths:.1f}"  + " & " + f"{ct5w_avg_reach_vuln_nodes:.1f}/100"  + \
           " & " +  f"{ct5w_avg_analysis_time:.1f} $\pm$ {ct5w_std_analysis_time:.1f}" + " \\\\")
    # print("\\textbf{Random Classifier} & " + f"{rc_cg_nodes:.1f} & {rc_cg_edges:.1f}" + " & " + f"{rc_avg_reach_vuln_paths:.1f}"  + " & " + \
    #        f"{rc_avg_reach_vuln_nodes:.1f}/100"  + " & " +  f"{rc_avg_analysis_time:.1f} $\pm$ {rc_std_analysis_time:.1f}" + " \\\\")


def main():
    parser = argparse.ArgumentParser(description='Process RQ argument.')
    parser.add_argument('rq', choices=['RQ1', 'RQ2', 'RQ2_1', 'RQ3', 'RQ3_1', 'RQ4'],
                        help='Specify which RQ to process')

    args = parser.parse_args()

    if args.rq == "RQ1":
        gen_RQ1_results()
    elif args.rq == "RQ2":
        gen_RQ2_results()
    elif args.rq == "RQ2_1":
        gen_RQ2_1_results()
    elif args.rq == "RQ3":
        gen_RQ3_results()
    elif args.rq == "RQ3_1":
        gen_RQ3_1_results()
    elif args.rq == "RQ4":
        gen_RQ4_results()

if __name__ == "__main__":
    main()
        