#!/bin/bash -l
#SBATCH --job-name=PIPELINE
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com
#SBATCH --output=slurm-%j.out

set -eo pipefail

# ==========================================
# CONFIGURATION
# Uncomment one block below (Smoke Test OR Real Run)
# ==========================================

# --- OPTION A: SMOKE TEST (Fast, for debugging) ---
# N1_PRE=500
# EPOCHS=2
# N2=200
# DENSITY=maf            # MAF is faster/more stable for very small samples
# REC_K=5
# REC_NUM_POST=500
# NUM_SAMPLES_INF=3      # Small number of posterior samples per subject

# --- OPTION B: REAL RUN (Production) ---
N1_PRE=30000
EPOCHS=8
N2=20000
DENSITY=nsf              # NSF is generally more expressive for complex posteriors
REC_K=200
REC_NUM_POST=4000
NUM_SAMPLES_INF=14988    # High number of posterior samples

# --- DATA PATHS ---
# Set folder to process real data (leave empty "" to skip Stage 4)
SST_FOLDER="/home/wwang/data/all_valid"
GLOB_PAT="NDAR_*.csv.zip"

# --- ENVIRONMENT ---
CONDA_ENV="pymc_env"

# ==========================================
# SETUP
# ==========================================

# 1. Threading Hygiene (limit thread contention)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# 2. Scratch Directory Setup
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch: ${SCRATCH_DIRECTORY}"

# 3. Copy source files to scratch
cd "${SCRATCH_DIRECTORY}"
cp "${SLURM_SUBMIT_DIR}/pomdp.py"                       .
cp "${SLURM_SUBMIT_DIR}/preprocessing.py"               . || true
cp "${SLURM_SUBMIT_DIR}/pipeline_transformer_snpe.py"   .

# Create output directory
mkdir -p outputs

# 4. Activate Environment
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "=== ENV INFO ==="
python -c "import sys,torch; print('Python', sys.version); print('Torch', torch.__version__)"

# ==========================================
# EXECUTION STAGES
# ==========================================

# --- Stage 1: Pretrain Encoder ---
echo "[Stage 1] Pretraining Encoder..."
srun --cpu-bind=cores python pipeline_transformer_snpe.py \
  --stage pretrain \
  --n1_pre ${N1_PRE} \
  --epochs ${EPOCHS}

# --- Stage 2: Train SNPE ---
echo "[Stage 2] Training SNPE (density=${DENSITY})..."
srun --cpu-bind=cores python pipeline_transformer_snpe.py \
  --stage snpe \
  --n1_pre ${N1_PRE} \
  --n2 ${N2} \
  --density ${DENSITY}

# --- Stage 3: Recovery Check ---
echo "[Stage 3] Running Recovery Sweep (K=${REC_K})..."
srun --cpu-bind=cores python pipeline_transformer_snpe.py \
  --stage recover \
  --K ${REC_K} \
  --num_post ${REC_NUM_POST}

# --- Stage 4: Inference on Real Data ---
if [[ -n "${SST_FOLDER}" && -d "${SST_FOLDER}" ]]; then
  echo "[Stage 4] Processing Real Data in: ${SST_FOLDER}"
  srun --cpu-bind=cores python pipeline_transformer_snpe.py \
    --stage posterior \
    --sst_folder "${SST_FOLDER}" \
    --glob_pat "${GLOB_PAT}" \
    --num_samples ${NUM_SAMPLES_INF}
else
  echo "[Stage 4] Skipped (SST_FOLDER not set or invalid)"
fi

# ==========================================
# SAVE RESULTS
# ==========================================

echo "Saving results to ${SLURM_SUBMIT_DIR}..."

if [ -d "outputs" ]; then
    cp -r outputs "${SLURM_SUBMIT_DIR}/"
    echo "Results saved."
else
    echo "Warning: outputs directory not found!"
fi

# ==========================================
# NO CLEANUP (As requested)
# ==========================================
# echo "Cleaning up scratch directory..."
# rm -rf "${SCRATCH_DIRECTORY}"

echo "Job completed successfully."
echo "Scratch directory preserved at: ${SCRATCH_DIRECTORY}"

exit 0