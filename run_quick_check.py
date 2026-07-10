#!/usr/bin/env python3
"""
run_quick_check.py

Automated Quick Run for the SST-POMDP Pipeline.
This script runs a lightweight, end-to-end test of the entire repository.
It supports both a DEMO mode (using example data) and a PROD mode (using local raw data).

Usage:
    python run_quick_check.py          # Runs in DEMO mode
    python run_quick_check.py --prod   # Runs in PROD mode
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# Use the current active Python executable (e.g., from the conda env)
PYTHON_EXE = sys.executable

# Automatically get the root directory of the repository (where this script lives)
REPO_ROOT = Path(__file__).resolve().parent

def run_command(step_name, script_path, extra_args=None, expected_output=None, prod_mode=False):
    """Helper function to run a command, check for missing scripts, track time, and handle errors."""
    
    script_file = REPO_ROOT / script_path
    
    # 1. Skip if the script itself hasn't been created yet
    if not script_file.exists():
        print(f"\n{'='*60}")
        print(f"SKIPPED: {step_name}")
        print(f"Reason: Script '{script_path}' does not exist yet.")
        print(f"{'='*60}")
        return True

    # 2. Skip if the outputs already exist (Caching to save time)
    if expected_output:
        # Convert single string to list for uniform processing
        if isinstance(expected_output, str):
            expected_output = [expected_output]
            
        # Check if ALL expected outputs exist
        all_exist = True
        for out_path in expected_output:
            if not (REPO_ROOT / out_path).exists():
                all_exist = False
                break
                
        if all_exist:
            print(f"\n{'='*60}")
            print(f"SKIPPED: {step_name}")
            print("Reason: All expected output files already exist.")
            print(f"{'='*60}")
            return True

    # Build the command string
    command_list = [PYTHON_EXE, str(script_file)]
    
    # Pass the --prod flag to child scripts if activated
    if prod_mode:
        command_list.append('--prod')
        
    if extra_args:
        command_list.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"CMD:  {' '.join(command_list)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Run the command and force it to execute inside the REPO_ROOT directory
        subprocess.run(command_list, check=True, text=True, cwd=REPO_ROOT)
        elapsed_time = time.time() - start_time
        print(f"\nSUCCESS: {step_name} completed in {elapsed_time:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\nFAILED: {step_name} failed after {elapsed_time:.2f} seconds.")
        print(f"Return code: {e.returncode}")
        print("Aborting the quick run.")
        sys.exit(1)

# =========================================================================
# QUICK CHECK SETTINGS
# =========================================================================
# The default model configuration to test in the quick run (can be changed anytime)
TEST_MODEL_NAME = "tesbi_e2e_6p_v1"

def main():
    # ---------------------------------------------------------
    # ARGUMENT PARSING
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Automated Quick Run for the SST-POMDP Pipeline")
    parser.add_argument(
        '--prod', 
        action='store_true', 
        help='Run entire pipeline in production mode with raw data'
    )
    args = parser.parse_args()

    print("Starting SST-POMDP Automated Quick Run...")
    if args.prod:
        print(">>> MODE: PRODUCTION (Local Raw Data) <<<")
    else:
        print(">>> MODE: DEMO (Example Processed Data) <<<")
        
    print(f"Repository Root Directory: {REPO_ROOT}")
    print("This will test the model dynamics, run a lightweight inference pipeline, and generate analysis figures.")
    
    # ---------------------------------------------------------
    # Step 1: Model Sanity Check
    # ---------------------------------------------------------
    run_command(
        "Model Sanity Check (fig_dynamics.py)",
        "analysis/fig_dynamics.py",
        expected_output=[
            "outputs_example/fig_action_values.png",
            "outputs_example/fig_belief_states.png",
            "outputs_example/fig_stop_prior_sensitivity.png"
        ],
        prod_mode=args.prod
    )

    run_command(
        "Model Sanity Check (fig_policy.py)",
        "analysis/fig_policy.py",
        expected_output=[
            "outputs_example/fig_policy_summary.png",
            "outputs_example/fig_policy_GS.png",
            "outputs_example/fig_policy_GE.png",
            "outputs_example/fig_policy_GM.png",
            "outputs_example/fig_policy_SS.png",
            "outputs_example/fig_policy_SE.png"
        ],
        prod_mode=args.prod
    )

    # ---------------------------------------------------------
    # Step 2: TeSBI HPC Workflow (Lightweight Quick Run)
    # ---------------------------------------------------------
    tesbi_data_folder = "./data/example_processed_data"
    filter_csv_path = "./data/example_clinical_behavior.csv"
    out_base_dir = "outputs_example/tesbi"

    model_suffix = TEST_MODEL_NAME.split("e2e_")[-1] if "e2e_" in TEST_MODEL_NAME else TEST_MODEL_NAME
    
    expected_weights_name = f"test_amortized_inference_net_{model_suffix}.pth"
    expected_weights_path = f"{out_base_dir}/{expected_weights_name}"

    # --- 2.1 TeSBI Model Fitting (Adaptive) ---
    tesbi_args = [
        "--stage", "all",
        "--n_sims", "10", 
        "--epochs", "2", 
        "--batch_size", "4", 
        "--K", "5", 
        "--num_post", "10", 
        "--sst_folder", tesbi_data_folder
    ]
    
    run_command(
        f"2.1 TeSBI Model Fitting ({TEST_MODEL_NAME})", 
        f"hpc/test_{TEST_MODEL_NAME}.py", 
        extra_args=tesbi_args,
        expected_output=expected_weights_path
    )

    # --- 2.2 Embedding Extraction Check ---
    embed_out_csv = f"{out_base_dir}/example_embeddings_{model_suffix}.csv"
    emb_args = [
        "--csv", filter_csv_path,
        "--folder", tesbi_data_folder,
        "--model_path", expected_weights_path,
        "--out", embed_out_csv
    ]
    
    run_command(
        "2.2 Embedding Extraction Check", 
        "hpc/eval_embed.py",
        extra_args=emb_args,
        expected_output=embed_out_csv
    )

    # ---------------------------------------------------------
    # Step 3: Results Analysis & Figure Generation
    # ---------------------------------------------------------
    # List of tuples: (Step Name, Script Path, Expected Output File(s))
    analysis_scripts = [
        ("Analysis: Parameter Distributions", "analysis/fig_param_dist.py", ["outputs_example/fig_param_dist.png"]),
        ("Analysis: ADHD vs Params", "analysis/fig_param_adhd.py", ["outputs_example/fig_param_adhd.png"]),
        ("Analysis: Canonical Correlation Analysis", "analysis/fig_cca.py", ["outputs_example/fig_cca.png"]),
        ("Analysis: PPC Representative Subjects", "analysis/fig_ppc_reps.py", ["outputs_example/fig_ppc_reps.png"]),
        ("Analysis: PCA Embedding", "analysis/fig_pca_embed.py", ["outputs_example/fig_pca_embed_2d.png"]),
        ("Analysis: Demographics Counts Table", "analysis/tab_demo_counts.py", ["outputs_example/tab_demo_counts.tex"]),
        ("Analysis: Linear Models (Behavior)", "analysis/tab_lm_behavior.py", ["outputs_example/tab_lm_behavior.tex"]),
        ("Analysis: Linear Models (Params)", "analysis/tab_lm_params.py", [
            "outputs_example/tab_lm_params_main.tex", 
            "outputs_example/tab_lm_params_supp.tex"
        ]),
        ("Analysis: Model Comparison", "analysis/tab_model_compare.py", [
            "outputs_example/tesbi/table1_specs.tex",
            "outputs_example/tesbi/table2_ppc.tex",
            "outputs_example/tesbi/table3_recovery.tex",
            "outputs_example/tesbi/table4_cv.tex"
        ]),
        ("Analysis: Parameter Quantiles", "analysis/tab_param_quantiles.py", ["outputs_example/tab_param_quantiles.csv"])
    ]

    for step_name, script_path, exp_out in analysis_scripts:
        run_command(step_name, script_path, expected_output=exp_out, prod_mode=args.prod)

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------
    print(f"\n{'*'*60}")
    print("ALL TESTS PROCESSED SUCCESSFULLY!")
    print("Check the 'outputs_example/' folder for the generated figures, tables, and models.")
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    main()