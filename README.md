# Computational Modeling of Cognitive Control in the Stop Signal Task (SST)

This repository contains the full data processing, Simulation-Based Inference (SBI) modeling, and analysis pipeline for fitting a Partially Observable Markov Decision Process (POMDP) to Stop Signal Task (SST) data, focusing on the relationship between model parameters and clinical traits (e.g., ADHD).

## Project Structure and File Descriptions

The project is divided into logical stages, ensuring a clean and repeatable workflow from raw data to final tables and figures.

### I. Data Preparation, Cleaning, and Core Modules

These scripts handle the foundational work: filtering raw data, calculating behavioral metrics, and defining the core cognitive model.

| Script | Purpose |
| :--- | :--- |
| **`filter.py`** | Filters the ABCD SST behavioral dataset based on quality control criteria (e.g., performance flags, trial counts) and copies the corresponding valid raw data files into a clean destination folder. |
| **`preprocessing.py`** | Core utility for data cleaning. Loads raw SST trial-level data, categorizes trials into outcomes (GS, GE, GM, SS, SE), discretizes timestamps (RT/SSD) into 25ms steps, and formats the data for model input. |
| **`stats.py`** | Calculates descriptive behavioral statistics (SSRT, reaction times) from raw SST files and merges them with demographic, CBCL (ADHD scores), medication, and IQ data to create a comprehensive analysis dataset. |
| **`metrics.py`** | Contains mathematical functions to calculate behavioral statistics (e.g., SSRT) and distance metrics (Wasserstein, Kolmogorov-Smirnov) used to quantify the discrepancy between observed human data and model simulations. |
| **`pomdp.py`** | Defines the **core cognitive model**, a Partially Observable Markov Decision Process (POMDP), for the Stop Signal Task (SST). It includes the mathematical structure, model initialization (perceptual noise, precision and cost parameters), dynamic decision policy calculation (value iteration), and behavioral simulation logic. |

---

### II. Simulation-Based Inference (SBI) Pipeline

These files manage the high-performance computing (HPC) steps for parameter estimation using a Transformer network as the summary statistic.

| Script | Purpose |
| :--- | :--- |
| **`pipeline_transformer_snpe.py`** | The **end-to-end SBI pipeline**. It trains a Transformer encoder to summarize trial-level SST data into low-dimensional embeddings, then trains a Neural Spline Flow (NSF) to approximate the posterior distribution of model parameters given these embeddings. It supports pretraining, active learning loops, parameter recovery checks, and final inference. |
| **`embedding.py`** | Generates subject-level embeddings by processing raw SST data through the pre-trained `TrialTransformer` encoder. It extracts the mean-pooled embedding vector (the summary statistic for SBI) and calculates secondary metrics like the standard deviation of the trial-level latents. |
| **`run_pipeline.sh`** | A robust SLURM batch script for running the full SBI pipeline on an HPC cluster. It handles scratch management and sequentially executes the pretraining, SNPE training, parameter recovery, and subject inference stages. |
| **`run_embedding.sh`** | A SLURM batch script for running `embedding.py` to generate subject embeddings on the cluster. It manages the HPC environment and retrieves the output. |

---

### III. Model Validation and Fit Analysis

These scripts validate the reliability of the parameter estimation and assess the model's goodness-of-fit to the data.

| Script | Purpose |
| :--- | :--- |
| **`ppc.py`** | Performs the core **Posterior Predictive Checks (PPC)** simulation step. It simulates the cognitive model using inferred MAP parameters for each subject and computes discrepancy metrics against observed data. |
| **`run_ppc.sh`** | A SLURM batch script to parallelize the execution of the `ppc.py` simulation on an HPC cluster. |
| **`recovery_analysis.py`** | Verifies the reliability of the parameter estimation pipeline. It plots "Ground Truth" vs. "Recovered" parameters and calculates correlation coefficients from the recovery study results. |
| **`ppc_popu_analysis.py`** | Visualizes the **global model fit across the entire population**. It loads the PPC discrepancy metrics and generates a grid of histograms showing the distribution of these fit errors for each metric. |
| **`ppc_analysis.py`** | Selects three **representative subjects** (Best, Median, and Worst fit) based on a composite fit cost. This script is used to visualize detailed individual subject fits (RT distributions, inhibition functions, outcome rates). |
| **`model_fit_analysis.py`** | Visualizes the overall model fit performance by plotting observed vs. simulated behavior for representative subjects, providing a detailed visual diagnostic of model fidelity. |

---

### IV. Parameter and Statistical Analysis

These scripts analyze the inferred parameters and generate final summary tables.

| Script | Purpose |
| :--- | :--- |
| **`param_analysis.py`** | Loads the inferred model parameters (MAP estimates), computes summary statistics (mean, median, range), and generates distribution plots (histograms with KDE overlays) for the key cognitive parameters. |
| **`param_lmm.py`** | Performs statistical analysis using Linear Mixed-Effects Models (LMM) to quantify the relationship between inferred cognitive parameters and clinical traits (e.g., ADHD), controlling for covariates (Sex, IQ, Medication). Outputs a formatted LaTeX table of the results. |
| **`count_demo.py`** | Generates a demographic summary table (LaTeX format) displaying subject counts stratified by key groups (ADHD score, Sex, IQ tertiles, Medication status). |

---

### V. Proposed Folder Structure for Organization

For improved clarity, modularity, and long-term maintenance, it is highly recommended to arrange the files into a functional subfolder structure.

```
sst-pomdp/
├── data/                             # Raw and processed external data
│   ├── raw_abcd_sst/                 # Original raw files (e.g., zips)
│   └── sst_valid_raw/                # Cleaned, filtered raw files (used by the pipeline)
|
├── src/                              # All Python code for the pipeline
│   ├── model_core/                   # Core math and model definition
│   │   ├── pomdp.py                  # POMDP model definition
│   │   └── metrics.py                # Behavioral and distance metrics
│   │
│   ├── data_prep/                    # Data cleaning and feature extraction
│   │   ├── filter.py                 # QC filtering
│   │   ├── stats.py                  # Behavioral stats calculation
│   │   └── preprocessing.py          # Trial-level cleaning/discretization
│   │
│   ├── sbi_inference/                # SBI (Transformer + SNPE) components
│   │   ├── pipeline_transformer_snpe.py # End-to-end SBI logic
│   │   └── embedding.py              # Subject embedding extraction
│   │
│   └── analysis/                     # Parameter analysis, LMMs, and PPC plotting
│       ├── param_analysis.py         # Parameter distribution plots
│       ├── param_lmm.py              # LMM for parameter-trait correlation
│       ├── count_demo.py             # Demographics table generation
│       ├── ppc_analysis.py           # Representative subject fit plots
│       └── ppc_popu_analysis.py      # Population PPC histograms
|
├── artifacts/                        # Intermediate/final model outputs (e.g., trained models, posteriors)
│   └── sbi_txf/                      # Transformer/SBI specific artifacts
│       ├── encoder.pt                # Trained Transformer encoder
│       └── posterior_summary_all.csv # Inferred parameter means for all subjects
|
└── scripts/                          # All HPC/SLURM and utility scripts
    ├── run_pipeline.sh               # Executes the full SBI pipeline on HPC
    ├── run_embedding.sh              # Executes embedding generation on HPC
    ├── run_ppc.sh                    # Executes PPC simulation on HPC
    └── README.md                     # Project documentation
```
