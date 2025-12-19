#!/bin/bash
#SBATCH --job-name=c121
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=20:02:00
#SBATCH --mem=4G
#SBATCH --array=0-999
#SBATCH --output=../slurm_log/%x_%A_%a.out
#SBATCH --constraint=amd

set -e

module purge              
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd ../input

# File containing list of input scripts
input_list="input_list.txt"

# Read the correct line based on SLURM_ARRAY_TASK_ID
inp=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$input_list")

# Safety check
if [ -z "$inp" ]; then
    echo "Error: No input found for array ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Prepare log directories
# mkdir -p ../log ../slurm_log
basename_inp=$(basename "$inp")
log="../log/${basename_inp%.lammps}.log"

echo "Running on $(hostname)"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Input file: $inp"
echo "Log file: $log"

# Run LAMMPS
srun $HOME/.local/bin/lmp_d9_double_aocc -in "$inp" > "$log"
