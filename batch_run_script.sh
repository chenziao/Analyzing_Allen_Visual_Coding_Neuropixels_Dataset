#!/bin/bash

#SBATCH -J analyze_allen_visual_coding_script
#SBATCH -o ./stdout/run_script%j.out
#SBATCH -e ./stdout/run_script%j.error
#SBATCH -t 0-12:00:00

#SBATCH -N 1
#SBATCH -n 1
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G


START=$(date)

unset DISPLAY
python scripts/find_probe_channels.py --session_set all --cache_data_only

END=$(date)

echo "Started running at $START."
echo "Finished running at $END."
