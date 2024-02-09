"""
Measuring WALA run-time
"""

from os.path import join, basename, exists
import subprocess
import time
import statistics
import re
import numpy as np

NJR1_FOLDER = "/mnt/data/amir_data/njr-1-dataset/june2020_dataset"
TEST_PROGRAMS = "/mnt/data/amir_projects/ml4cg/data/njr1/test_programs.txt"
WALA_CG_GEN_JAR = "/mnt/data/amir_projects/ml4cg/ml4cg_SA/target/method_extractor-1.0-SNAPSHOT-shaded.jar"
OUTPUT_FOLDR = "/mnt/data/amir_projects/ml4cg/src/scripts/tmp"

def parse_log_file(filename):
    # Pattern to match lines containing "in xxx.xx seconds"
    time_p = r'in (\d+\.\d+) seconds'
    program_p = r'for(.*?)in \d+(\.\d+)? seconds'
    times = []
    processed_progams = []

    with open(filename, 'r') as file:
        for line in file:
            if "Successfully generated a call graph for" in line:
                m = re.search(program_p, line)
                p = m.group(1).strip() if m else None
                if p not in processed_progams:
                    processed_progams.append(p)
                    match = re.search(time_p, line)
                    if match:
                        # Convert the matched time to a float and append it to the list
                        times.append(float(match.group(1)))

    # Convert the list of times to a numpy array
    times = np.array(times)

    print(f"No. of programs {len(times)}")
    # Compute and print the mean and standard deviation of the times
    print(f'Mean: {np.mean(times):.2f}')
    print(f'Standard deviation: {np.std(times):.2f}')


# processed_progams = []
# with open('/mnt/data/amir_projects/ml4cg/src/scripts/logs/run_wala', 'r') as f:
#     pattern = r'\burl.*?\b'
#     for l in f.readlines():
#         #m = re.findall(pattern, l)
#         if "Successfully generated a call graph for" in l:
#             m = re.split(r'\bin\b', l)
#             #p = m[0] if m else None
#             p = m[0].strip() if m else None
#             processed_progams.append(p.replace("Successfully generated a call graph for", " ").strip())
        

def benchmark_wala_cg_gen():
    runtime_wala_njr1 = []
    with open(TEST_PROGRAMS, 'r') as f:
        test_programs_l = [l.rstrip() for l in f.readlines()]
        for p in test_programs_l:
            p_jar = join(NJR1_FOLDER, p, "jarfile", p + ".jar")
        # if p not in processed_progams:
            print("JAR file:", p_jar)
            command = f'java -cp {WALA_CG_GEN_JAR} dev.c0pslab.analysis.WalaCGGen -o {OUTPUT_FOLDR} -f {p_jar}'
            s_t = time.time()
            result = subprocess.run(command, shell=True, check=False)
            e_t = time.time() - s_t
            runtime_wala_njr1.append(e_t)
            if result.returncode == 0:
                print(f"Successfully generated a call graph for {p} in {e_t:.2f} seconds")
            else:
                print(f"Failed to generate a call graph for {p}")
        # else:
        #     print(f"CG for {p} arleady exsits!")
            
        print(f"WALA runtime: {statistics.mean(runtime_wala_njr1):.2f} seconds per test program on average")


if __name__ == "__main__":
    benchmark_wala_cg_gen()
    #parse_log_file('/mnt/data/amir_projects/ml4cg/src/scripts/logs/run_wala_3')
