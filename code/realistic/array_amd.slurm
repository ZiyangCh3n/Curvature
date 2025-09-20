#!/bin/bash
#SBATCH --array=501-1000
#SBATCH --nodes=1
#SBATCH --job-name=c047_2
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --output=../slurm_log/%A_%a.out
#SBATCH --constraint=amd

# Load modules
module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Define parameter arrays
a_values=("cur" "str")
d_values=(2.5)
b_values=(1.0)
c_values=({0..999})

# Compute lengths
len_a=${#a_values[@]}
len_d=${#d_values[@]}
len_b=${#b_values[@]}
len_c=${#c_values[@]}
total_combinations=$((len_a * len_d * len_b * len_c))

# Calculate task range
batch_size=$SLURM_NTASKS_PER_NODE
start_index=$(( (SLURM_ARRAY_TASK_ID - 1) * batch_size ))
end_index=$(( start_index + batch_size - 1 ))

if [ $end_index -ge $total_combinations ]; then
  end_index=$((total_combinations - 1))
fi

# Move to input directory
cd ../input

# Pre-log all commands that this array task will run
echo "[$(date)] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID will run the following commands:" >&2
for ((i = start_index; i <= end_index; i++)); do
  a_index=$((i / (len_d * len_b * len_c)))
  d_index=$(((i % (len_d * len_b * len_c)) / (len_b * len_c)))
  b_index=$(((i % (len_b * len_c)) / len_c))
  c_index=$((i % len_c))

  a=${a_values[$a_index]}
  d=${d_values[$d_index]}
  b=${b_values[$b_index]}
  c=${c_values[$c_index]}

  input_file="./${a}_${d}_${b}_0.5_1.2_${c}.lammps"
  log_file="../log/${a}_${d}_${b}_0.5_1.2_${c}.log"
  echo "srun $HOME/.local/bin/lmp_d9_double_aocc -in $input_file > $log_file" >&2
done
echo "[$(date)] Begin running jobs below..." >&2

# Launch LAMMPS jobs in background
for ((i = start_index; i <= end_index; i++)); do
  a_index=$((i / (len_d * len_b * len_c)))
  d_index=$(((i % (len_d * len_b * len_c)) / (len_b * len_c)))
  b_index=$(((i % (len_b * len_c)) / len_c))
  c_index=$((i % len_c))

  a=${a_values[$a_index]}
  d=${d_values[$d_index]}
  b=${b_values[$b_index]}
  c=${c_values[$c_index]}

  input_file="./${a}_${d}_${b}_0.5_1.2_${c}.lammps"
  log_file="../log/${a}_${d}_${b}_0.5_1.2_${c}.log"
  srun $HOME/.local/bin/lmp_d9_double_aocc -in $input_file > $log_file &

  if (( (i - start_index + 1) % batch_size == 0 )); then
    wait
  fi
done

wait
