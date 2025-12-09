#!/bin/bash
# ==============================================================================
# SLURM Array Job Submission
# ==============================================================================
# Splits sessions across multiple parallel tasks for faster processing.
# Automatically combines logs after all tasks complete.
#
# Usage: Edit the configuration below, then run:
#   bash submit_array_job.sh
#
# Output:
#   - Combined log: output/batch_logs/<script>_YYYYMMDD_HHMMSS.log
#   - SLURM output: stdout/run_script<JOB_ID>_<TASK_ID>.out
#
# Note: Individual task logs are stored temporarily in output/batch_logs/<script>_job<ID>/
#       and deleted after combining.
# ==============================================================================

# ===================== Configuration - Edit These =====================
# Number of parallel array tasks
NUM_TASKS=10

# Python script to run
SCRIPT_PATH="scripts/find_probe_channels.py"

# Arguments passed to the script
SCRIPT_ARGS="--session_set all"
# ======================================================================

SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)
ARRAY_MAX=$((NUM_TASKS - 1))

mkdir -p stdout

echo "============================================"
echo "Submitting SLURM Array Job"
echo "  Script: $SCRIPT_PATH"
echo "  Tasks: $NUM_TASKS"
echo "  Args: $SCRIPT_ARGS"
echo "  Time: $(date)"
echo "============================================"

# Submit main array job
MAIN_JOB=$(sbatch --parsable --array=0-${ARRAY_MAX} <<EOF
#!/bin/bash
#SBATCH -J ${SCRIPT_NAME}_array
#SBATCH -o ./stdout/run_script%A_%a.out
#SBATCH -e ./stdout/run_script%A_%a.error
#SBATCH -t 0-12:00:00
#SBATCH -N 1 -n 1
#SBATCH --mem-per-cpu=16G

START=\$(date)
echo "Array task \$SLURM_ARRAY_TASK_ID/\$SLURM_ARRAY_TASK_COUNT on \$SLURMD_NODENAME"

unset DISPLAY
python $SCRIPT_PATH $SCRIPT_ARGS --array_index \$SLURM_ARRAY_TASK_ID --array_total $NUM_TASKS

END=\$(date)
echo ""
echo "Array tasks started at \$START"
echo "Array tasks finished at \$END"

EOF
)

echo "Main job submitted: $MAIN_JOB"

# Submit combine job (runs after all array tasks complete)
COMBINE_JOB=$(sbatch --parsable --dependency=afterok:${MAIN_JOB} <<EOF
#!/bin/bash
#SBATCH -J ${SCRIPT_NAME}_combine_logs
#SBATCH -o ./stdout/combine_%j.out
#SBATCH -e ./stdout/combine_%j.error
#SBATCH -t 0-00:05:00
#SBATCH -N 1 -n 1
#SBATCH --mem-per-cpu=1G

START=\$(date)
echo "Combining logs from job $MAIN_JOB"

python scripts/combine_array_logs.py "${SCRIPT_NAME}_job${MAIN_JOB}"

END=\$(date)
echo ""
echo "Combine logs started at \$START"
echo "Combine logs finished at \$END"

EOF
)

echo "Combine job submitted: $COMBINE_JOB (runs after $MAIN_JOB completes)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $MAIN_JOB"
