#!/bin/bash
#SBATCH --array=1-3              # Job array of size 5
#SBATCH --job-name=rerun
#SBATCH --nodes=1
#SBATCH --ntasks=32                 # 10 parallel tasks
#SBATCH --cpus-per-task=2           # Each task uses 2 CPUs
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=01:02:00
#SBATCH --output=../slurm_log/%x_%A_%a.out
#SBATCH --error=../slurm_log/%x_%A_%a.err
#SBATCH --constraint=amd

# Load modules
module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

# Set OpenMP threads per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd ../input

INPUT_LIST="../code/rerun.txt" # make sure newline at end of file
CHUNK_SIZE=32
START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END_LINE=$(( START_LINE + CHUNK_SIZE - 1 ))
TOTAL_LINES=$(wc -l < "$INPUT_LIST")

if [ "$END_LINE" -gt "$TOTAL_LINES" ]; then
    END_LINE=$TOTAL_LINES
fi

echo "Array task $SLURM_ARRAY_TASK_ID processing lines $START_LINE to $END_LINE..."

# Bounds checking
if [ "$START_LINE" -gt "$TOTAL_LINES" ]; then
    echo "No input files for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi



# Read chunk of input files
mapfile -t FILES_TO_PROCESS < <(sed -n "${START_LINE},${END_LINE}p" "$INPUT_LIST")

for INPUT_FILE in "${FILES_TO_PROCESS[@]}"; do
    FILE_NAME=$(basename "$INPUT_FILE")
    BASE_NAME="${FILE_NAME%.*}"
    LOG_FILE="../log/${BASE_NAME}.log"

    echo "Running: $INPUT_FILE -> $LOG_FILE"
    $HOME/.local/bin/lmp_d9_double_aocc -in "$INPUT_FILE" > "$LOG_FILE" 2>&1 &
done

wait
echo "Array task $SLURM_ARRAY_TASK_ID completed."
