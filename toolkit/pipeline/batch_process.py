import json
import argparse
import inspect
import datetime
import sys
import os

import contextlib
import traceback
from pathlib import Path

from dataclasses import dataclass

from ..paths import paths
from ..pipeline.global_settings import GLOBAL_SETTINGS
from .data_io import SessionSet, get_sessions, safe_mkdir

from typing import Sequence, Callable, Any


# Parse arguments from command line

@dataclass
class Parameter():
    name: str
    default: Any
    type: type | str = None
    help: str = ""

    def __str__(self) -> str:
        """Get the help text for the parameter."""
        help = self.help.rstrip(' .')
        if help:
            help += ". "
        help += f"Default: {self.default}."
        return help


def parameters_from_dict(parameters_dict: dict[str, dict[str, Any]]) -> list[Parameter]:
    parameters = []
    for name, kwargs in parameters_dict.items():
        parameters.append(Parameter(name, **kwargs))
    return parameters


def split_sessions_for_array(sessions: list[int], array_index: int, array_total: int) -> list[int]:
    """Split sessions into groups for SLURM array jobs."""
    from numpy import array_split
    splits = array_split(sessions, array_total)
    return splits[array_index].tolist() if array_index < len(splits) else []


class BatchProcessArgumentParser():

    def __init__(self,
        parameters: Sequence[Parameter] | dict[str, dict[str, Any]],
        session_set: SessionSet | str | list[int] | None = None,
    ):
        """Create an argument parser for the batch process script.

        Parameters
        ----------
        default_parameters: dict[str, Any]
            The dictionary of parameters and default values for the process function.
        parameters_type: dict[str, type | str] = None
            The types of the parameters. If None, the type will be inferred from the default value.
        parameters_help: dict[str, str] = None
            The help text for the parameters. If None, the help text will be only the default value.
        session_set: SessionSet | str | list[int] | None
            The default session set to process the sessions from.
            If not provided, the session set will be retrieved from the global settings.
        """
        if isinstance(parameters, dict):
            parameters = parameters_from_dict(parameters)
        self.parameters = {param.name: param for param in parameters}
        self.default_parameters = {param.name: param.default for param in parameters}

        if session_set is None:
            session_set = GLOBAL_SETTINGS.get('session_set', 'all')
        available_session_sets = [s.name for s in SessionSet if s != SessionSet.CUSTOM]

        # Create argument parser
        self.parser = parser = argparse.ArgumentParser()
        parser.add_argument(
            '--session_set', type=str, default=session_set,
            help=f"The session set to process the sessions from. Available sets: {', '.join(available_session_sets)}."
        )
        parser.add_argument(
            '--session_list', type=int, nargs='+', default=[],
            help="List of session IDs to process (space-separated). "
                "'--session_set' argument will be ignored if this is provided."
        )
        parser.add_argument(
            '--use_blacklist', action='store_true', default=False,
            help="Use sessions blacklist to exclude sessions to process."
        )
        # SLURM array job support
        parser.add_argument(
            '--array_index', type=int, default=None,
            help="SLURM array task index (0-based). Use with --array_total for parallel processing. "
                "If SLURM_ARRAY_TASK_ID env var is set, it will be used automatically."
        )
        parser.add_argument(
            '--array_total', type=int, default=None,
            help="Total number of SLURM array tasks. Sessions will be split evenly across tasks."
        )
        self.add_parameter_to_parser()
        parser.add_argument(
            '--disable_logging', action='store_true', default=False,
            help="Disable logging to the log file."
        )

    def add_parameter_to_parser(self):
        """Add parameters to the argument parser."""
        parser = self.parser
        for param, default in self.default_parameters.items():
            parameter = self.parameters.get(param)
            arg_type = parameter.type
            if arg_type is None:
                arg_type = type(default)
            elif isinstance(arg_type, str):
                arg_type = eval(arg_type)
            help = str(parameter)  # Get help text for the parameter

            match arg_type.__name__:
                case 'bool':
                    group = parser.add_mutually_exclusive_group()
                    group.add_argument(f'--{param}', action='store_true', dest=param, help=help)
                    group.add_argument(f'--no-{param}', action='store_false', dest=param, help=f"Disable {param}")
                    parser.set_defaults(**{param: default})
                case 'list':
                    parser.add_argument(
                        f'--{param}', type=str, nargs='+', default=default,
                        help=f"{help} Provide list input as space-separated values."
                    )
                case _:
                    try:
                        parser.add_argument(f'--{param}', type=arg_type, default=default, help=help)
                    except Exception as e:
                        raise TypeError(f"Invalid argument type for parameter '{param}': '{arg_type.__name__}'") from e

    def parse_args(self) -> dict[str, Any]:
        """Parse the arguments and return the dictionary of arguments for the `process_sessions()` function."""
        args = self.parser.parse_args()
        parameters = {key: getattr(args, key) for key in self.default_parameters.keys()}
        
        # Auto-detect SLURM array job from environment variables if not provided
        array_index = args.array_index
        array_total = args.array_total
        if array_index is None and 'SLURM_ARRAY_TASK_ID' in os.environ:
            array_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if array_total is None and 'SLURM_ARRAY_TASK_COUNT' in os.environ:
            array_total = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        
        arguments = dict(
            parameters=parameters,
            session_set=args.session_list if args.session_list else args.session_set,
            use_blacklist=args.use_blacklist,
            disable_logging=args.disable_logging,
            array_index=array_index,
            array_total=array_total
        )
        return arguments


# Process sessions

def get_blacklist_sessions() -> list[int]:
    """Get the blacklist sessions from the sessions file."""
    with open(paths.SESSIONS_FILE, 'r') as f:
        sessions_config = json.load(f)
    return sessions_config.get('blacklist', [])


def filter_blacklist_sessions(sessions: list[int]) -> list[int]:
    """Filter the sessions with blacklist and return the accepted and blacklisted sessions."""
    blacklist_sessions = set(get_blacklist_sessions())
    accepted_sessions = []
    blacklisted_sessions = []
    for session in sessions:
        if session in blacklist_sessions:
            blacklisted_sessions.append(session)
        else:
            accepted_sessions.append(session)
    return accepted_sessions, blacklisted_sessions


def get_timestamp(format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Get the current timestamp in local time in the given format."""
    return datetime.datetime.now().astimezone().strftime(format)


def get_log_info(process_function: Callable) -> str:
    """Get the log file for the process function with format <script_name>_<timestamp>[_array<N>]
    
    Parameters
    ----------
    process_function : Callable
        The process function to get the log info for.
    array_suffix : str
        Optional suffix for array job identification (e.g., "_array0").
    """
    try:
        script_name = Path(inspect.getfile(process_function)).stem
    except TypeError:
        script_name = process_function.__name__
    return script_name, get_timestamp("%Y%m%d_%H%M%S")


def log_file_path(script_name: str, suffix: str, array_job_id: str = None) -> Path:
    """Get the path for a log file.
    
    For array jobs, logs are stored in a subdirectory named by SLURM_JOB_ID to 
    avoid conflicts between different runs.
    """
    batch_log_dir = paths.BATCH_LOG_DIR
    if array_job_id:
        batch_log_dir = batch_log_dir / f"{script_name}_job{array_job_id}"
    safe_mkdir(batch_log_dir)
    return batch_log_dir / f"{script_name}_{suffix}.log"


def parameters_file_path(script_name: str, timestamp: str) -> Path:
    """Get the path for the parameters JSON file."""
    safe_mkdir(paths.BATCH_LOG_DIR)
    return paths.BATCH_LOG_DIR / f"{script_name}_parameters_{timestamp}.json"


class TeeOutput:
    """A class that writes to both a file and the original stdout/stderr."""
    def __init__(self, file, original_stream):
        self.file = file
        self.original_stream = original_stream
    
    def write(self, text):
        # Write to both the file and the original stream
        self.file.write(text)
        self.file.flush()  # Ensure it's written to file immediately
        self.original_stream.write(text)
        self.original_stream.flush()  # Ensure it's written to console immediately
    
    def flush(self):
        self.file.flush()
        self.original_stream.flush()
    
    def __getattr__(self, name):
        # Delegate any other attributes to the original stream
        return getattr(self.original_stream, name)


def write_to_log(file, tee_output=True):
    """Context manager to write output to the log file.
    
    Parameters
    ----------
    file : file-like object
        The file to write the log to
    tee_output : bool, default True
        If True, output will be written to both the file and the original stdout/stderr.
        If False, output will only be written to the file.
    """
    @contextlib.contextmanager
    def _log():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            if tee_output:
                # Create tee objects that write to both file and original streams
                sys.stdout = TeeOutput(file, old_stdout)
                sys.stderr = TeeOutput(file, old_stderr)
            else:
                # Original behavior - only write to file
                sys.stdout = file
                sys.stderr = file
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _log()


@contextlib.contextmanager
def print_to_file(file_path, tee_output=True, disable=False, **kwargs):
    """Context manager that opens a file and redirects print() statements to it.
    
    Parameters
    ----------
    file_path : str or Path
        The path to the file to write to
    tee_output : bool, default True
        If True, output will be written to both the file and the original stdout/stderr.
        If False, output will only be written to the file.
    disable : bool, default False
        If True, output will not be written to the file.
    kwargs : dict
        Keyword arguments to pass to the function open(file_path, 'w', **kwargs).
    """
    if disable:
        print(f"Logging is disabled (not writing to file: {file_path})\n")
        yield None
    else:
        kwargs = {'mode': 'w', **kwargs}  # Default to write mode
        print(f"Logging to file: {file_path}\n")
        with open(file_path, **kwargs) as f:
            with write_to_log(f, tee_output=tee_output):
                yield f


def process_sessions(
    process_function: Callable[[int, ...], Any],
    parameters: dict[str, Any],
    session_set: SessionSet | str | list[int],
    use_blacklist: bool = False,
    disable_logging: bool = False,
    array_index: int = None,
    array_total: int = None
):
    """Process sessions with the given function.
    
    Supports three execution modes:
    - Local/single job: All sessions processed sequentially, single log file.
    - SLURM array job: Sessions split across tasks, logs in job-specific directory.
    
    Parameters
    ----------
    process_function : Callable
        The process function to execute for each session. First argument must be session ID.
    parameters : dict[str, Any]
        The dictionary of parameters to pass to the process function.
    session_set : SessionSet | str | list[int]
        Session set to process (e.g., 'all', 'selected', or list of IDs).
    use_blacklist : bool
        If True, exclude blacklisted sessions.
    disable_logging : bool
        If True, don't write to log file.
    array_index : int, optional
        SLURM array task index (0-based). Auto-detected from environment.
    array_total : int, optional
        Total number of array tasks. Auto-detected from environment.
    """
    sessions, session_set = get_sessions(session_set)
    if use_blacklist:
        sessions, blacklist_sessions = filter_blacklist_sessions(sessions)
    
    # Determine if running as array job
    is_array_job = array_index is not None and array_total is not None
    array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID') if is_array_job else None
    
    # Sessions to process in this task
    if is_array_job:
        task_sessions = split_sessions_for_array(sessions, array_index, array_total)
    else:
        task_sessions = sessions
    
    # Setup logging paths
    script_name, timestamp = get_log_info(process_function)
    combined_log_file = log_file_path(script_name, timestamp)
    if is_array_job:
        log_file = log_file_path(script_name,
            suffix=f"{timestamp}_array{array_index}", array_job_id=array_job_id)
    else:
        log_file = combined_log_file

    # For first task of array job or non-array job
    if not is_array_job or array_index == 0:
        # Save parameters
        params_file = parameters_file_path(script_name, timestamp)
        with open(params_file, 'w', encoding='utf-8') as pf:
            json.dump(parameters, pf, indent=4)

        # Write header to combined log file 
        with print_to_file(combined_log_file, encoding='utf-8', disable=disable_logging) as f:
            # Function and arguments 
            print(f"Process script: {script_name}")
            print(f"Process function: {process_function.__name__}")
            print("Parameters:")
            for key, value in parameters.items():
                print(f"  {key}: {value}")

            # Session set and session ids
            print(f"\nProcessing session set: {session_set.name}")
            print("Session IDs:")
            for session in sessions:
                print(f"  {session}")

            if use_blacklist:
                print("\nBlacklisted sessions:")
                for session in blacklist_sessions:
                    print(f"  {session}")

            print("\n" + "*" * 80 + "\n")

    # Write processing log
    aborted_sessions = []
    mode = 'w' if is_array_job else 'a'  # Append to combined file for non-array job
    with print_to_file(log_file, encoding='utf-8', disable=disable_logging, mode=mode) as f:
        # Start processing
        if is_array_job:
            print(f"Array task {array_index + 1}/{array_total}")
        print(f"\nProcessing started at {get_timestamp()}")
        print("\n" + "=" * 80 + "\n")

        # Main processing loop
        for session in task_sessions:
            print(f"\nProcessing session: {session} at {get_timestamp()}")
            print("-" * 80 + "\n")
            try:
                process_function(session, **parameters)
            except Exception as e:
                aborted_sessions.append(session)
                print(f"\n\n[ERROR] Exception occurred while processing session {session}:\n")
                traceback.print_exc()
                print("\n\n" + "-" * 80)
                print(f"Session {session} aborted at {get_timestamp()}\n")
            else:
                print("\n\n" + "-" * 80)
                print(f"Session {session} processed successfully at {get_timestamp()}\n")

        # End processing
        print("\n" + "=" * 80)
        print(f"\nProcessing completed at {get_timestamp()}")

        # Print task summary
        n_success = len(task_sessions) - len(aborted_sessions)
        print(f"\nNumber of successfully processed sessions: {n_success}/{len(task_sessions)}")
        if aborted_sessions:
            print(f"Aborted sessions: {' '.join(map(str, aborted_sessions))}")


def combine_array_logs(job_dir: str | Path, delete: bool = True) -> Path | None:
    """Combine array job log files into the combined log file (which already has the header).
    
    Appends individual task logs to the header file and aggregates aborted sessions
    from all tasks as a summary at the end.
    
    Parameters
    ----------
    job_dir : str | Path
        Path to the array job log directory (e.g., "find_probe_channels_job12345").
        Can be absolute path or relative to BATCH_LOG_DIR.
    delete : bool
        If True, delete the job directory after combining.
        
    Returns
    -------
    Path | None
        Path to the combined log file, or None if no files found.
    """
    import re
    import shutil
    
    job_dir = Path(job_dir)
    if not job_dir.is_absolute():
        job_dir = paths.BATCH_LOG_DIR / job_dir
    
    if not job_dir.is_dir():
        print(f"Directory not found: {job_dir}")
        return None
    
    log_files = sorted(job_dir.glob("*.log"))
    if not log_files:
        print(f"No log files in: {job_dir}")
        return None
    
    # Output file: header file already exists (scriptname_timestamp.log)
    output_name = log_files[0].stem.rsplit("_array", 1)[0] + ".log"
    output_path = paths.BATCH_LOG_DIR / output_name
    
    # Parse aborted sessions from each task log
    all_aborted = []
    total_success = 0
    total_sessions = 0
    aborted_pattern = re.compile(r"Aborted sessions: (.+)")
    success_pattern = re.compile(r"Number of successfully processed sessions: (\d+)/(\d+)")
    
    with open(output_path, 'a', encoding='utf-8') as out:
        for f in log_files:
            content = f.read_text(encoding='utf-8')
            out.write(f"\n{'#' * 30} {f.name} {'#' * 30}\n\n")
            out.write(content)
            
            # Extract aborted sessions (space-separated)
            match = aborted_pattern.search(content)
            if match and match.group(1):
                aborted_ids = [int(x) for x in match.group(1).split()]
                all_aborted.extend(aborted_ids)
            
            # Extract success counts
            match = success_pattern.search(content)
            if match:
                total_success += int(match.group(1))
                total_sessions += int(match.group(2))
        
        # Write aggregated summary
        out.write(f"\n\n{'=' * 80}\n")
        out.write(f"BATCH SUMMARY\n")
        out.write(f"{'=' * 80}\n")
        out.write(f"Total sessions processed: {total_sessions}\n")
        out.write(f"Total successful: {total_success}\n")
        out.write(f"Total aborted: {len(all_aborted)}\n")
        if all_aborted:
            out.write(f"All aborted sessions: {' '.join(map(str, all_aborted))}\n")
    
    print(f"Combined {len(log_files)} files -> {output_path}")
    
    if delete:
        shutil.rmtree(job_dir)
        print(f"Deleted: {job_dir}")
    
    return output_path
