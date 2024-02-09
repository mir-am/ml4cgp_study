# On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study

This repository contains the code and data to replicate the results of our emprical study on "the Effectiveness of Machine Learning-based Call Graph Pruning", which is accepted at the technical track of the MSR'24 conference.

# Abstract
Static call graph (CG) construction often over-approximates call relations, leading to sound, but imprecise results. Recent research
has explored machine learning (ML)-based CG pruning as a means
to enhance precision by eliminating false edges. However, current
methods suffer from a limited evaluation dataset, imbalanced train-
ing data, and reduced recall, which affects practical downstream
analyses. Prior results were also not compared with advanced static
CG construction techniques yet. This study tackles these issues. We
introduce the NYXCorpus, a dataset of real-world Java programs
with high test coverage and we collect traces from test executions
and build a ground truth of dynamic CGs. We leverage these CGs
to explore conservative pruning strategies during the training and
inference of ML-based CG pruners. The study compares 0-CFA-
based static CGs with a context-sensitive 1-CFA algorithm, both
with and without pruning. We find that CG pruning is a difficult
task for real-world Java projects and substantial improvements in
the CG precision (+25%) meet reduced recall (-9%). However, our ex-
periments show promising results: even when we favor recall over
precision by using an F2 metric in our experiments, we can show
that pruned CGs have comparable quality to a context-sensitive
1-CFA analysis while being computationally less demanding. Re-
sulting CGs are much smaller (69%), and substantially faster (3.5x
speed-up), with virtually unchanged results in our downstream
analysis.

# Installation 
## Requirements
### Software 
- Linux-based OS (Ubuntu 22.04 LTS)
- Python 3.9 or newer
- Java 11 and Maven
- Nvidia CUDA toolkit 11.8

### Hardware
| Hardware   | Minimum      | Recommended  |
|------------|--------------|--------------|
| CPU        | 8-Core CPU (AMD Ryzen 7 7700X)| 24-Core CPU (Intel Core i9 13900K) |
| RAM        | 32GB         | 64GB or higher        |
| GPU        | RTX 3080 12GB | RTX 4090 24GB |
| Storage    | 512GB SSD    | 1TB NVME SSD    |


## Quick install
Before running the experiments, install the project and its dependencies first.

```
$ git clone https://github.com/mir-am/ml4cgp_study && cd ml4cgp_study
pip install .
mvn -f ml4cg_SA/pom.xml install
```

# Data & Models
For this empirical study, the datasets including `NYXCorpus` and fine-tuned code language models for call graph pruning can be downloaded from Zenodo.

# Usage
To replicate the results of the research questions in the paper, run the following command:

```
$ ML4CGP_DATA=$PATH sh run_experiments.sh RQ1
```
Replace `$PATH` to the dataset folder you downloaded and extracted from Zenodo.
You can also replace `RQ1` with `RQ2`, `RQ3` or `RQ4` to run the experiments for other RQs.
