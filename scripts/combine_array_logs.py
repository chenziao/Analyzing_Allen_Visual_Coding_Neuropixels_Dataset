#!/usr/bin/env python
"""Combine array job logs from a job directory.

Usage:
    python scripts/combine_array_logs.py <job_dir> [--keep]
    
Examples:
    python scripts/combine_array_logs.py find_probe_channels_job12345
    python scripts/combine_array_logs.py find_probe_channels_job12345 --keep
"""
import add_path
import sys
from toolkit.pipeline.batch_process import combine_array_logs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    job_dir = sys.argv[1]
    delete = "--keep" not in sys.argv
    combine_array_logs(job_dir, delete=delete)
