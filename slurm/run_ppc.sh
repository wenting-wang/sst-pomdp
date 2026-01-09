#!/bin/bash -l

#SBATCH --job-name=PPC_SIM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=20-00:00:00
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenting.wang.cd@outlook.com

# --- Configuration ---
ENV_NAME="pymc_env"
SCRIPT_NAME="ppc.py"
OUTPUT_FILE="ppc_metrics.csv" 
# Note: Ensure this OUTPUT_FILE name matches the 'OUTPUT_PPC_CSV' filename defined in ppc.py

# Exit immediately if a command exits with a non-zero status
set -e

# --- Setup Scratch Directory ---
# Define and create a unique scratch directory
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
echo "Running in scratch directory: ${SCRATCH_DIRECTORY}"

cd ${SCRATCH_DIRECTORY}

# --- Copy Dependencies ---
# Copy all necessary python scripts to the scratch location
cp ${SLURM_SUBMIT_DIR}/models.py .
cp ${SLURM_SUBMIT_DIR}/simulation.py .
cp ${SLURM_SUBMIT_DIR}/metrics.py .
cp ${SLURM_SUBMIT_DIR}/preprocessing.py .
cp ${SLURM_SUBMIT_DIR}/${SCRIPT_NAME} .

# --- Environment Setup ---
# Activate Anaconda environment
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# --- Execution ---
echo "Starting simulation at $(date)"
python ${SCRIPT_NAME}
echo "Simulation finished at $(date)"

# --- Save Results ---
# Copy the output CSV back to the directory where the job was submitted
# Note: This assumes ppc.py writes to the current directory (scratch). 
# If ppc.py writes to an absolute path (like /home/user/...), this line is redundant but harmless.
if [ -f "${OUTPUT_FILE}" ]; then
    cp ${OUTPUT_FILE} ${SLURM_SUBMIT_DIR}/
    echo "Results copied to ${SLURM_SUBMIT_DIR}"
else
    echo "Warning: Output file ${OUTPUT_FILE} not found in scratch. Check if python script wrote to a different path."
fi

# --- Cleanup ---
# Clean up scratch directory to free up space
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
echo "Scratch directory cleaned up."

exit 0