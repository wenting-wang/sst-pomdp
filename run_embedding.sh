#!/bin/bash -l
#SBATCH --job-name=GEN_EMBED
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_embed-%j.out

set -eo pipefail

# ==========================================
# CONFIGURATION
# ==========================================
SST_FOLDER="/home/wwang/data/all_valid"      # Data source
CSV_FILE="stats_all_cbcl_demo.csv"           # Subject list (in submit dir)
OUTPUT_FILE="embeddings_mean_std.csv"        # Output filename

# Absolute path to your existing trained model
SOURCE_MODEL_PATH="/home/wwang/ptmp/823879/artifacts_sbi_txf/encoder.pt"

CONDA_ENV="pymc_env"

# ==========================================
# SETUP
# ==========================================

# 1. Threading Hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# 2. Scratch Directory Setup
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch: ${SCRATCH_DIRECTORY}"

# 3. Copy files to scratch
cd "${SCRATCH_DIRECTORY}"

# A) Copy scripts and CSV from the directory where you submitted the job
cp "${SLURM_SUBMIT_DIR}/embedding.py"       .
cp "${SLURM_SUBMIT_DIR}/preprocessing.py"   . || true
cp "${SLURM_SUBMIT_DIR}/${CSV_FILE}"        .

# B) Copy the trained model from your specific path to here
#    (Using the filename 'encoder.pt' in scratch)
echo "Copying model from ${SOURCE_MODEL_PATH}..."
cp "${SOURCE_MODEL_PATH}" ./encoder.pt

# 4. Activate Environment
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# ==========================================
# EXECUTION
# ==========================================

echo "Starting Embedding Generation..."

# Note: We pass --model_path "encoder.pt" because we copied it to the current folder (.)
srun python embedding.py \
    --csv "${CSV_FILE}" \
    --folder "${SST_FOLDER}" \
    --model_path "encoder.pt" \
    --out "${OUTPUT_FILE}"

# ==========================================
# SAVE RESULTS
# ==========================================

echo "Copying result (${OUTPUT_FILE}) back to submit directory..."
cp "${OUTPUT_FILE}" "${SLURM_SUBMIT_DIR}/"

echo "Job Complete."