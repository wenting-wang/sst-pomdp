#!/bin/bash -l
#SBATCH --job-name=TeSBI_CV_Batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive                  
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=slurm_cv_batch-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module purge
module load rocm/7.0
module load python-waterboa/2025.06
source ~/rocm_env/bin/activate  

# ==============================================================================
# Configuration
# ==============================================================================
SST_FOLDER="/u/wenwang/data/sst_valid_base" 
FILTER_CSV="/u/wenwang/data/clinical_behavior.csv"

# Define the models to evaluate. Format: "config_name|absolute_pth_path"
# Separate multiple models with spaces or newlines.
MODELS=(
    "tesbi_e2e_10p_v1|/u/wenwang/my_ptmp/10293580/outputs/amortized_inference_net_10p_v1.pth"
    "tesbi_e2e_8p_v1|/u/wenwang/my_ptmp/10299815/outputs/amortized_inference_net_8p_v1.pth"
    "tesbi_e2e_8p_v2|/u/wenwang/my_ptmp/10302317/outputs/amortized_inference_net_8p_v2.pth"
    "tesbi_e2e_8p_v3|/u/wenwang/my_ptmp/10303122/outputs/amortized_inference_net_8p_v3.pth"
    "tesbi_e2e_8p_v4|/u/wenwang/my_ptmp/10303497/outputs/amortized_inference_net_8p_v4.pth"
    "tesbi_e2e_8p_v5|/u/wenwang/my_ptmp/10304067/outputs/amortized_inference_net_8p_v5.pth"
    "tesbi_e2e_7p_v1|/u/wenwang/my_ptmp/10314618/outputs/amortized_inference_net_7p_v1.pth"
    "tesbi_e2e_7p_v2|/u/wenwang/my_ptmp/10314798/outputs/amortized_inference_net_7p_v2.pth"
    "tesbi_e2e_7p_v3|/u/wenwang/my_ptmp/10315091/outputs/amortized_inference_net_7p_v3.pth"
    "tesbi_e2e_7p_v4|/u/wenwang/my_ptmp/10315258/outputs/amortized_inference_net_7p_v4.pth"
    "tesbi_e2e_7p_v5|/u/wenwang/my_ptmp/10315397/outputs/amortized_inference_net_7p_v5.pth"
    "tesbi_e2e_7p_v6|/u/wenwang/my_ptmp/10315600/outputs/amortized_inference_net_7p_v6.pth"
    "tesbi_e2e_6p_v1|/u/wenwang/my_ptmp/10331248/outputs/amortized_inference_net_6p_v1.pth"
    "tesbi_e2e_6p_v2|/u/wenwang/my_ptmp/10331334/outputs/amortized_inference_net_6p_v2.pth"
    "tesbi_e2e_6p_v3|/u/wenwang/my_ptmp/10331343/outputs/amortized_inference_net_6p_v3.pth"
)

# ==============================================================================
# Execution Loop
# ==============================================================================
echo "================================================================"
echo "Starting Batch Out-of-Sample CV Evaluation..."
echo "================================================================"

mkdir -p outputs

for ENTRY in "${MODELS[@]}"; do
    CONFIG_NAME="${ENTRY%%|*}"
    WEIGHT_PATH="${ENTRY##*|}"
    
    echo -e "\n----------------------------------------------------------------"
    echo ">> Evaluating Model Configuration: ${CONFIG_NAME}"
    echo ">> Weights Loading From: ${WEIGHT_PATH}"
    echo "----------------------------------------------------------------"
    
    if [ ! -f "${WEIGHT_PATH}" ]; then
        echo "[Warning] Weights file not found at ${WEIGHT_PATH}. Skipping..."
        continue
    fi

    srun python3 -u cv_out_of_sample.py \
        --sst_folder "${SST_FOLDER}" \
        --filter_csv "${FILTER_CSV}" \
        --model_config "${CONFIG_NAME}" \
        --weights_path "${WEIGHT_PATH}" \
        --train_ratio 0.8
done

echo -e "\n================================================================"
echo "All Models Evaluated. Please check the ./outputs/ directory for summary CSVs."
echo "================================================================"