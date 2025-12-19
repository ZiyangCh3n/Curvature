#!/bin/bash
#SBATCH --job-name=gel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:02:00
#SBATCH --mem=4G
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

# Run LAMMPS
srun $HOME/.local/bin/lmp_d9_double_aocc -in gel_step1.lammps > gel_step1.log
srun $HOME/.local/bin/lmp_d9_double_aocc -in gel_step2.lammps > gel_step2.log

