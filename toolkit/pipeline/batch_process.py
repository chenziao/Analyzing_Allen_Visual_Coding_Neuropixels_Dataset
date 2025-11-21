import json
import argparse
import inspect
import datetime
import sys
import contextlib
import traceback
from pathlib import Path

from dataclasses import dataclass

from ..paths import paths
from ..pipeline.global_settings import GLOBAL_SETTINGS
from .data_io import SessionSet, get_sessions

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
        arguments = dict(
            parameters=parameters,
            session_set=args.session_list if args.session_list else args.session_set,
            use_blacklist=args.use_blacklist,
            disable_logging=args.disable_logging
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
    """Get the log file for the process function with format <script_name>_<timestamp>"""
    try:
        script_name = Path(inspect.getfile(process_function)).stem
    except TypeError:
        script_name = process_function.__name__
    return script_name, get_timestamp("%Y%m%d_%H%M%S")


def log_file_path(script_name: str, timestamp: str) -> Path:
    batch_log_dir = paths.BATCH_LOG_DIR
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    return batch_log_dir / f"{script_name}_{timestamp}.log"


def parameters_file_path(script_name: str, timestamp: str) -> Path:
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
        print(f"Logging to file: {file_path}\n")
        with open(file_path, 'w', **kwargs) as f:
            with write_to_log(f, tee_output=tee_output):
                yield f


def process_sessions(
    process_function: Callable[[int, ...], Any],
    parameters: dict[str, Any],
    session_set: SessionSet | str | list[int],
    use_blacklist: bool = False,
    disable_logging: bool = False
):
    """Process the sessions from the session set.

    Parameters
    ----------
    process_function: Callable
        The process function to execute for each session. The first argument must be the session ID.
    parameters: dict[str, Any]
        The dictionary of parameters to pass to the process function.
    session_set: SessionSet | str | list[int]
        The session set to process the sessions from.
    use_blacklist: bool = False,
        If True, the blacklist sessions will be excluded from the session set.
    disable_logging: bool = False
        If True, logging will not be written to the log file.

    Returns
    -------
    None
    """
    sessions, session_set = get_sessions(session_set)
    if use_blacklist:
        sessions, blacklist_sessions = filter_blacklist_sessions(sessions)
    aborted_sessions = []

    # Get log file and parameters file
    script_name, timestamp = get_log_info(process_function)
    log_file = log_file_path(script_name, timestamp)
    parameters_file = parameters_file_path(script_name, timestamp)

    # Write to log file
    with print_to_file(log_file, encoding='utf-8', disable=disable_logging) as f:
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

        # Write parameters to file
        with open(parameters_file, 'w', encoding='utf-8') as pf:
            json.dump(parameters, pf, indent=4)

        # Start processing
        print(f"\n\nProcessing started at {get_timestamp()}")
        print("\n" + "=" * 80 + "\n")

        # Main processing loop
        for session in sessions:
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

        # Print aborted sessions
        print(f"\nNumber of successfully processed sessions: {len(sessions) - len(aborted_sessions)}")
        if aborted_sessions:
            print(f"Number of aborted sessions: {len(aborted_sessions)}")
            print("\nAborted sessions:")
            for session in aborted_sessions:
                print(f"  {session}")
