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
SST_FOLDER="/home/wwang/data/sst_valid_base"     # Data source (Needs to match CLEAN_DATA_DIR in filter.py)
CSV_FILE="stats_beh_full.csv"                   # Subject list (In submit dir)
OUTPUT_FILE="embeddings.csv"                    # Output filename
OUTPUT_DIR="${SLURM_SUBMIT_DIR}"                # Final destination for results

# Absolute path to your existing trained model (encoder.pt)
# NOTE: This path must exist before the job starts.
SOURCE_MODEL_PATH="/home/wwang/ptmp/833725/outputs/encoder.pt"

CONDA_ENV="pymc_env"

# ==========================================
# SETUP
# ==========================================

# 1. Threading Hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# 2. Scratch Directory Setup
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch: ${SCRATCH_DIRECTORY}"

# 3. Copy files to scratch
cd "${SCRATCH_DIRECTORY}"

# A) Copy scripts and metadata CSV from the directory where you submitted the job
cp "${SLURM_SUBMIT_DIR}/embedding.py"       .
cp -r "${SLURM_SUBMIT_DIR}/utils/"   .
cp "${SLURM_SUBMIT_DIR}/${CSV_FILE}"        .

# B) Copy the trained model artifact and rename it locally to 'encoder.pt'
echo "Copying model from ${SOURCE_MODEL_PATH}..."
cp "${SOURCE_MODEL_PATH}" ./encoder.pt

# 4. Activate Environment
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# ==========================================
# EXECUTION
# ==========================================

echo "Starting Embedding Generation..."

# Run the python script
# We pass the metadata CSV, the raw data folder, the encoder path (now local 'encoder.pt'), and the output filename.
srun python embedding.py \
    --csv "${CSV_FILE}" \
    --folder "${SST_FOLDER}" \
    --model_path "./encoder.pt" \
    --out "${OUTPUT_FILE}"

# ==========================================
# SAVE RESULTS & NO CLEANUP
# ==========================================

echo "Saving results to ${OUTPUT_DIR}..."

# Copy the output CSV back to the submit directory
if [ -f "${OUTPUT_FILE}" ]; then
    cp "${OUTPUT_FILE}" "${OUTPUT_DIR}/"
    echo "Results copied successfully."
else
    echo "ERROR: Output file ${OUTPUT_FILE} not found in scratch."
fi

# Cleanup is skipped as requested, but we print the location
echo "Scratch directory preserved at: ${SCRATCH_DIRECTORY}"

exit 0