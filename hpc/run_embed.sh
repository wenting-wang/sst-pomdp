#!/bin/bash -l
#SBATCH --job-name=TeSBI_Emb
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  
#SBATCH --mem=64G                     
#SBATCH --cpus-per-task=16            
#SBATCH --time=12:00:00
#SBATCH --output=slurm_embedding-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com

set -eo pipefail

# ==============================================================================
# 1. PIPELINE CONFIGURATION
# ==============================================================================
MODEL_FILE="tesbi_e2e_6p_v1"

SST_FOLDER="/u/wenwang/data/sst_valid_base" 
FILTER_CSV="/u/wenwang/data/clinical_behavior.csv"
WEIGHTS_PATH="/u/wenwang/my_ptmp/10331248/outputs/amortized_inference_net_6p_v1.pth"

OUT_CSV="embeddings_e2e_${MODEL_FILE}.csv"

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "================================================================"

module purge
module load rocm/7.0
module load python-waterboa/2025.06
source ~/rocm_env/bin/activate  

echo "Python Environment Activated. Checking AMD ROCm GPU Status:"
rocm-smi || echo "rocm-smi not available, moving on."

# ==============================================================================
# 3. SCRATCH SPACE I/O HANDLING
# ==============================================================================
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOB_ID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Staging files to scratch directory: ${SCRATCH_DIRECTORY}"

cp "${SLURM_SUBMIT_DIR}/embedding.py" "${SCRATCH_DIRECTORY}/"
cp "${SLURM_SUBMIT_DIR}/${MODEL_FILE}.py" "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/utils/" "${SCRATCH_DIRECTORY}/"
cp -r "${SLURM_SUBMIT_DIR}/core/"                "${SCRATCH_DIRECTORY}/"

if [ -f "${WEIGHTS_PATH}" ]; then
    cp "${WEIGHTS_PATH}" "${SCRATCH_DIRECTORY}/"
else
    echo "ERROR: Model weights not found at ${WEIGHTS_PATH}"
    exit 1
fi

cd "${SCRATCH_DIRECTORY}"

# ==============================================================================
# 4. EXECUTE THE EMBEDDING EXTRACTION
# ==============================================================================
JOB_START_TIME=$(date +%s)
echo -e "\n================================================================"
echo ">>> STEP 1: Running Embedding Extraction at $(date)"
echo "================================================================"

WEIGHT_FILENAME=$(basename "${WEIGHTS_PATH}")

srun python3 -u embedding.py \
    --csv "${FILTER_CSV}" \
    --folder "${SST_FOLDER}" \
    --model_path "${WEIGHT_FILENAME}" \
    --out "${OUT_CSV}"

# ==============================================================================
# 5. RETRIEVE RESULTS
# ==============================================================================
JOB_END_TIME=$(date +%s)
echo -e "\n================================================================"
echo ">>> Copying results back to permanent storage..."

FINAL_OUT_DIR="${SLURM_SUBMIT_DIR}/outputs_embedding_${SLURM_JOB_ID}"
mkdir -p "${FINAL_OUT_DIR}"

if [ -f "${OUT_CSV}" ]; then
    cp "${OUT_CSV}" "${FINAL_OUT_DIR}/"
    echo "Result successfully copied to: ${FINAL_OUT_DIR}/${OUT_CSV}"
else
    echo "WARNING: Expected output file '${OUT_CSV}' not found in scratch!"
fi

total_duration=$((JOB_END_TIME - JOB_START_TIME))
echo "================================================================"
echo "Job Completed Successfully."
echo "Total Runtime: $(($total_duration / 3600)) hrs $((($total_duration % 3600) / 60)) min $(($total_duration % 60)) sec."
echo "================================================================"
exit 0