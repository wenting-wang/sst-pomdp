#!/bin/bash -l
#SBATCH --job-name=TeSBI
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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
# N1_PRE=100
# EPOCHS=5
# PATIENCE=1           
# N2=100
# DENSITY=maf            
# REC_K=10
# REC_NUM_POST=100
# NUM_SAMPLES_INF=100   

# --- OPTION B: REAL RUN (Production) ---
N1_PRE=30000
EPOCHS=100
PATIENCE=15           
N2=20000
DENSITY=nsf              
REC_K=500
REC_NUM_POST=2000
NUM_SAMPLES_INF=2000    

# # --- DATA PATHS ---
SST_FOLDER="/home/wwang/data/sst_valid_base"
GLOB_PAT="NDAR_*.csv.zip"

# # ==========================================
# # SETUP
# # ==========================================

# # 1. Threading Hygiene 
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# # 2. Scratch Directory Setup
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch: ${SCRATCH_DIRECTORY}"

# # 3. Copy source files to scratch
cd "${SCRATCH_DIRECTORY}"
cp -r "${SLURM_SUBMIT_DIR}/core/"    .
cp -r "${SLURM_SUBMIT_DIR}/utils/"   .
cp "${SLURM_SUBMIT_DIR}/tesbi_m0.py"    .
# cp "${SLURM_SUBMIT_DIR}/check_gpu.py" . # Optional

mkdir -p outputs

# 4. Activate Environment
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate pymc_env

unset CUDA_VISIBLE_DEVICES
echo "Running nvidia-smi to confirm hardware visibility:"
nvidia-smi

# ==========================================
# EXECUTION STAGES
# ==========================================

JOB_START_TIME=$(date +%s)

# --- Stage 1: Pretrain Encoder ---
echo "----------------------------------------------------------------"
echo "[Stage 1] Pretraining Encoder... Started at $(date)"
SECONDS=0

srun python -u tesbi_m0.py \
  --stage pretrain \
  --n1_pre ${N1_PRE} \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE}

duration=$SECONDS
echo "[Stage 1] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."

# --- Stage 2: Train SNPE ---
echo "----------------------------------------------------------------"
echo "[Stage 2] Training SNPE... Started at $(date)"
SECONDS=0

srun python -u tesbi_m0.py \
  --stage snpe \
  --n1_pre ${N1_PRE} \
  --n2 ${N2} \
  --density ${DENSITY}

duration=$SECONDS
echo "[Stage 2] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."

# --- Stage 3: Recovery Check ---
echo "----------------------------------------------------------------"
echo "[Stage 3] Recovery Sweep... Started at $(date)"
SECONDS=0

srun python -u tesbi_m0.py \
  --stage recover \
  --K ${REC_K} \
  --num_post ${REC_NUM_POST}

duration=$SECONDS
echo "[Stage 3] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."

# --- Stage 4: Inference on Real Data ---
echo "----------------------------------------------------------------"
echo "[Stage 4] Real Data Inference... Started at $(date)"
SECONDS=0

if [[ -n "${SST_FOLDER}" && -d "${SST_FOLDER}" ]]; then
  srun python -u tesbi_m0.py \
    --stage posterior \
    --sst_folder "${SST_FOLDER}" \
    --glob_pat "${GLOB_PAT}" \
    --num_samples ${NUM_SAMPLES_INF}
    
  duration=$SECONDS
  echo "[Stage 4] Finished. Duration: $(($duration / 60)) min $(($duration % 60)) sec."
else
  echo "[Stage 4] Skipped."
fi

# ==========================================
# FINAL SUMMARY
# ==========================================
JOB_END_TIME=$(date +%s)
total_duration=$((JOB_END_TIME - JOB_START_TIME))
echo "================================================================"

# echo "Saving results to ${SLURM_SUBMIT_DIR}..."
# if [ -d "outputs" ]; then
#     cp -r outputs "${SLURM_SUBMIT_DIR}/"
#     echo "Results saved."
# else
#     echo "Warning: outputs directory not found!"
# fi

echo "Job Finished. Results are NOT copied back."
echo "You can find them here: ${SCRATCH_DIRECTORY}"

echo "Job Completed Successfully."
echo "Total Runtime: $(($total_duration / 3600)) hrs $((($total_duration % 3600) / 60)) min $(($total_duration % 60)) sec."
exit 0