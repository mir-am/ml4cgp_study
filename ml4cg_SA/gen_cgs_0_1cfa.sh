#!/bin/bash

# CLI Args:
# $1 -> Path to the dataset

jar_file_path="target/ml4cg_sa-1.0-SNAPSHOT-shaded.jar"

#XCorpus
java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/xcorpus/xcorpus_jars_w_deps" -j "$1/xcorpus/xcorpus_sel_programs.txt" -0cfa
java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/xcorpus/xcorpus_jars_w_deps" -j "$1/xcorpus/xcorpus_sel_programs.txt" -1cfa

#YCorpus
java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/ycorpus/gh_projects_processed_w_deps_v7-5" -j "$1/ycorpus/ycorpus_sel_programs.txt" -0cfa
java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/ycorpus/gh_projects_processed_w_deps_v7-5" -j "$1/ycorpus/ycorpus_sel_programs.txt" -1cfa

#NJR-1
java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/njr1/cgs/0cfa" -j "$1/njr1/cgs/njr1_programs.txt" -0cfa
java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/njr1/cgs/1cfa" -j "$1/njr1/cgs/njr1_programs.txt" -1cfa