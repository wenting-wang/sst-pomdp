#!/bin/bash -l
#SBATCH --job-name=rdex_3sub
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_rdex_fit-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com

# Recommended environment variables by MPCDF (prevents node crashes from excessive threading)
# This is crucial for R's parallel package so OpenBLAS does not multiply the thread count.
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Scratch Directory Setup for fast I/O
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p "${SCRATCH_DIRECTORY}"
echo "Running in scratch directory: ${SCRATCH_DIRECTORY}"

cd "${SCRATCH_DIRECTORY}"

# Copy all required files to the scratch directory
echo ">>> Copying R script, data, and DMC framework to scratch..."

############## TEST
cp "${SLURM_SUBMIT_DIR}/rdex_hpc/rdex_fit_test.R" .
cp "${SLURM_SUBMIT_DIR}/rdex_hpc/NDAR_INV8KX9UXKE_dmc_ready.csv" .

# ############## RUN
# cp "${SLURM_SUBMIT_DIR}/rdex_hpc/rdex_fit.R" .
# cp "${SLURM_SUBMIT_DIR}/rdex_hpc/NDAR_INV8KX9UXKE_dmc_ready.csv" .
# cp "${SLURM_SUBMIT_DIR}/rdex_hpc/NDAR_INVK4H5ZTFE_dmc_ready.csv" .
# cp "${SLURM_SUBMIT_DIR}/rdex_hpc/NDAR_INVURJ3XK7W_dmc_ready.csv" .

cp -r "${SLURM_SUBMIT_DIR}/dmc/" .

# Load the official R environment
module purge
module load R  # You may need to specify the version, e.g., module load R/4.2.2 if required by Viper

# Run the R fitting script
echo ">>> Starting RDEX-ABCD DE-MCMC Fitting on CPU cores..."
# srun ensures proper resource tracking within the Slurm step
srun Rscript rdex_fit.R

# Copy results back to the original directory
echo ">>> Fitting finished. Copying RDS results back to the submit directory..."
cp *.RDS "${SLURM_SUBMIT_DIR}/"

# Clean up scratch space
cd "${SLURM_SUBMIT_DIR}"
rm -rf "${SCRATCH_DIRECTORY}"
echo ">>> Job complete and scratch directory cleaned."