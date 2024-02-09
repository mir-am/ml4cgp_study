import time
import subprocess

OUTPUT_FOLDER = "/mnt/data/amir_projects/ml4cg/data/high_tc_dataset/analysis/cgs/ycorpus_train_sel_programs_1.json"
INPUT_JARS_FILE = "/mnt/data/amir_projects/ml4cg/data/high_tc_dataset/analysis/ycorpus_train_sel_programs_1.txt"
STATIC_CG_GEN_JAR = "/mnt/data/amir_projects/ml4cg/ml4cg_SA/target/method_extractor-1.0-SNAPSHOT-shaded.jar"
JVM_ARGS = "-Xmx32g"

cg_algs = {'-bc': "Basic CHA", "-ce": "Extended CHA", "-0cfa": "Zero CFA", "-rta": "RTA"}

# ['-bc', '-ce', '-0cfa', '-rta']
for alg in ['-0cfa']:
    command = f'java {JVM_ARGS} -cp {STATIC_CG_GEN_JAR} dev.c0pslab.analysis.CGGenRunner -o {OUTPUT_FOLDER} -j {INPUT_JARS_FILE} {alg}'
    print(command)
    print(f"Using the {cg_algs[alg]}")
    s_t = time.time()
    result = subprocess.run(command, shell=True, check=False)
    e_t = time.time() - s_t
    print(f"Generated CGs with the {cg_algs[alg]} in {e_t} seconds")