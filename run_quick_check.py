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
            "outputs/fig_action_values.png",
            "outputs/fig_belief_states.png",
            "outputs/fig_stop_prior_effects.png"
        ],
        prod_mode=args.prod
    )

    run_command(
        "Model Sanity Check (fig_policy.py)",
        "analysis/fig_policy.py",
        expected_output=[
            "outputs/fig_policy_summary.png",
            "outputs/fig_policy_GS.png",
            "outputs/fig_policy_GE.png",
            "outputs/fig_policy_GM.png",
            "outputs/fig_policy_SS.png",
            "outputs/fig_policy_SE.png"
        ],
        prod_mode=args.prod
    )

    # ---------------------------------------------------------
    # Step 2: TeSBI HPC Workflow (Lightweight Quick Run)
    # ---------------------------------------------------------
    if args.prod:
        tesbi_data_folder = "/Users/w/Desktop/data/sst_valid_base"
        filter_csv_path = "/Users/w/sst-hdbm-mac/sst-pomdp-main/data/clinical_behavior.csv"
    else:
        tesbi_data_folder = "./data/example_processed_data"
        filter_csv_path = "./data/example_clinical_behavior.csv"

    # Determine the suffix to guess the weights path (e.g., "6p_v1")
    model_suffix = TEST_MODEL_NAME.split("e2e_")[-1] if "e2e_" in TEST_MODEL_NAME else TEST_MODEL_NAME
    expected_weights_name = f"amortized_inference_net_{model_suffix}.pth"
    expected_weights_path = f"outputs/{expected_weights_name}"

    # --- 2.1 End-to-End Model Fitting (Adaptive) ---
    tesbi_args = [
        "--stage", "all", 
        "--n_sims", "10",      # Very low for quick check
        "--epochs", "2",       # Very low for quick check
        "--batch_size", "4", 
        "--K", "5", 
        "--num_post", "10", 
        "--sst_folder", tesbi_data_folder
    ]
    
    run_command(
        f"2.1 Model Fitting E2E ({TEST_MODEL_NAME})", 
        f"hpc/{TEST_MODEL_NAME}.py", 
        extra_args=tesbi_args,
        expected_output=expected_weights_path
    )

    # --- 2.2 Out-of-Sample Cross Validation Check ---
    cv_args = [
        "--sst_folder", tesbi_data_folder,
        "--filter_csv", filter_csv_path,
        "--model_config", TEST_MODEL_NAME,
        "--weights_path", expected_weights_path,
        "--train_ratio", "0.8"
    ]
    
    run_command(
        "2.2 Cross-Validation Check", 
        "hpc/cv_out_of_sample.py", 
        extra_args=cv_args
    )

    # --- 2.3 Embedding Extraction Check ---
    # Note: Adjust --model_path if your embedding.py expects just the filename instead of a relative path.
    emb_args = [
        "--csv", filter_csv_path,
        "--folder", tesbi_data_folder,
        "--model_path", expected_weights_name, # or expected_weights_path based on embedding.py logic
        "--out", f"embeddings_quick_check_{TEST_MODEL_NAME}.csv"
    ]
    
    run_command(
        "2.3 Embedding Extraction Check", 
        "hpc/embedding.py", 
        extra_args=emb_args,
        expected_output=f"outputs/embeddings_quick_check_{TEST_MODEL_NAME}.csv"
    )

    # ---------------------------------------------------------
    # Step 3: Results Analysis & Figure Generation
    # ---------------------------------------------------------
    # List of tuples: (Step Name, Script Path, Expected Output File(s))
    analysis_scripts = [
        ("Analysis: Parameter Distributions", "analysis/fig_param_dist.py", ["outputs/fig_param_dist.png"]),
        ("Analysis: ADHD vs Params", "analysis/fig_param_adhd.py", ["outputs/fig_param_adhd.png"]),
        ("Analysis: PPC Representative Subjects", "analysis/fig_ppc_reps.py", ["outputs/fig_ppc_reps.png"]),
        ("Analysis: Canonical Correlation Analysis", "analysis/fig_cca.py", ["outputs/fig_cca.png"]),
        ("Analysis: PCA Embedding", "analysis/fig_pca_embed.py", ["outputs/fig_pca_embed_2d.png"]),
        ("Analysis: Demographics Counts Table", "analysis/tab_demo_counts.py", ["outputs/tab_demo_counts.tex"]),
        ("Analysis: Linear Models (Behavior)", "analysis/tab_lm_behavior.py", ["outputs/tab_lm_behavior.tex"]),
        ("Analysis: Linear Models (Params)", "analysis/tab_lm_params.py", ["outputs/tab_lm_params_main.tex"])
    ]

    for step_name, script_path, exp_out in analysis_scripts:
        run_command(step_name, script_path, expected_output=exp_out, prod_mode=args.prod)

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------
    print(f"\n{'*'*60}")
    print("ALL TESTS PROCESSED SUCCESSFULLY!")
    print("Check the 'outputs/' folder for the generated figures, tables, and models.")
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    main()