#!/bin/bash
# ==============================================================================
# SLURM Array Job Submission
# ==============================================================================
# Splits sessions across multiple parallel tasks for faster processing.
# Automatically combines logs after all tasks complete.
#
# Usage:
#   ./submit_array_job.sh [NUM_TASKS] [SCRIPT_ARGS...]
#
# Examples:
#   ./submit_array_job.sh 10                        # 10 parallel tasks
#   ./submit_array_job.sh 5 --session_set selected  # 5 tasks, selected sessions
#   ./submit_array_job.sh 8 --use_blacklist         # 8 tasks, skip blacklisted
#
# Output:
#   - Individual logs: output/batch_logs/<script>_job<ID>/*.log
#   - Combined log: output/batch_logs/<script>_YYYYMMDD_HHMMSS_combined.log
#   - SLURM output: stdout/run_<JOB_ID>_<TASK_ID>.out
# ==============================================================================

NUM_TASKS=${1:-10}
shift 2>/dev/null
SCRIPT_ARGS="$@"
ARRAY_MAX=$((NUM_TASKS - 1))

mkdir -p stdout

echo "============================================"
echo "Submitting SLURM Array Job"
echo "  Tasks: $NUM_TASKS"
echo "  Args: $SCRIPT_ARGS"
echo "============================================"

# Submit main array job
MAIN_JOB=$(sbatch --parsable --array=0-${ARRAY_MAX} <<EOF
#!/bin/bash
#SBATCH -J analyze_allen_visual_coding_script
#SBATCH -o ./stdout/run_script%A_%a.out
#SBATCH -e ./stdout/run_script%A_%a.error
#SBATCH -t 0-12:00:00
#SBATCH -N 1 -n 1
#SBATCH --mem-per-cpu=16G

echo "Array task \$SLURM_ARRAY_TASK_ID/\$SLURM_ARRAY_TASK_COUNT on \$SLURMD_NODENAME"

unset DISPLAY
python scripts/find_probe_channels.py --session_set all $SCRIPT_ARGS

EOF
)

echo "Main job submitted: $MAIN_JOB"

# Submit combine job (runs after all array tasks complete)
COMBINE_JOB=$(sbatch --parsable --dependency=afterok:${MAIN_JOB} <<EOF
#!/bin/bash
#SBATCH -J combine_logs
#SBATCH -o ./stdout/combine_%j.out
#SBATCH -e ./stdout/combine_%j.error
#SBATCH -t 0-00:05:00
#SBATCH -N 1 -n 1
#SBATCH --mem-per-cpu=1G

echo "Combining logs from job $MAIN_JOB"
python scripts/combine_array_logs.py "find_probe_channels_job${MAIN_JOB}"

EOF
)

echo "Combine job submitted: $COMBINE_JOB (runs after $MAIN_JOB completes)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $MAIN_JOB"
