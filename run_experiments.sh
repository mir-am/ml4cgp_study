#!/bin/bash

if [ $# -eq 0 ]
then
    echo "No arguments provided. Please provide an argument (RQ1, RQ2, RQ3, RQ4)."
    exit 1
fi

if [ -z "$ML4CG_DATA_PATH" ]; then
    echo "Error: ML4CG_DATA_PATH ENV. variable is not set. Please provide a valid path to the dataset folder."
    exit 1
fi

echo "ML4CG_DATA_PATH ENV. variable is set to: $ML4CG_DATA_PATH"

if [ -z "$TESTING_MODE" ]; then
    export TESTING_MODE="0"
else
    echo "Testing mode is activated!"
fi

case $1 in
    RQ1)
        echo "Running experiments for RQ1"
        python -m src.scripts.run_random_classifier
        python -m src.scripts.run_RF_model
        python -m src.scripts.run_models RQ1
        python -m src.scripts.generate_results RQ1
        ;;
    RQ2)
        echo "Running experiments for RQ2"
        python -m src.scripts.run_models RQ2
        python -m src.scripts.generate_results RQ2
        python -m src.scripts.generate_results RQ2_1
        ;;
    RQ3)
        echo "Running experiments for RQ3"
        python -m src.scripts.run_models RQ3
        python -m src.scripts.run_feat_ext_inf
        python -m src.scripts.generate_results RQ3
        # python -m src.scripts.generate_results RQ3_1
        ;;
    RQ4)
        echo "Running experiments for RQ4"
        python -m src.scripts.add_rand_vuln_cg_nodes
        java -cp ml4cg_SA/target/ml4cg_sa-1.0-SNAPSHOT-shaded.jar "dev.c0pslab.analysis.VulnerabilityAnalysis" -i "${ML4CG_DATA_PATH}nyx_dataset/nyx_corpus_cgs_w_vuln_nodes.json" -o "${ML4CG_DATA_PATH}results/nyx_corpus_vuln_analysis.json"
        python -m src.scripts.generate_results RQ4
        ;;
    *)
        echo "Invalid argument. Please use one of: RQ1, RQ2, RQ3, RQ4."
        exit 1
        ;;
esac
