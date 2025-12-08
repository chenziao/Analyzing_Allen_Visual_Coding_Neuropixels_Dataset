#!/bin/bash
# ==============================================================================
# Single SLURM Job Submission
# ==============================================================================
# Processes all sessions sequentially in a single job.
# Use this for smaller workloads or when parallelization is not needed.
#
# Usage:
#   sbatch batch_run_script.sh
#
# Output:
#   - Log file: output/batch_logs/find_probe_channels_YYYYMMDD_HHMMSS.log
#   - SLURM output: stdout/run_<JOB_ID>.out
# ==============================================================================

#SBATCH -J analyze_allen_visual_coding_script
#SBATCH -o ./stdout/run_script%j.out
#SBATCH -e ./stdout/run_script%j.error
#SBATCH -t 0-12:00:00

#SBATCH -N 1 -n 1
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

mkdir -p stdout

echo "Single job started at $(date)"
echo "Processing all sessions sequentially..."

unset DISPLAY
python scripts/find_probe_channels.py --session_set all

echo "Finished running at $(date)."
