import json
import argparse
import inspect
import datetime
import sys
import contextlib
import traceback
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from ..paths import paths
from ..pipeline.global_settings import GLOBAL_SETTINGS

from typing import Sequence, Callable, Any


class SessionSet(Enum):
    ALL = 'all'
    TEST = 'test'
    SELECTED = 'selected'
    OPTOTAG = 'optotag'
    CUSTOM = 'custom'


def get_sessions(session_set : SessionSet | str | list[int]) -> tuple[list[int], SessionSet]:
    """Get the sessions from the session set.

    Parameters
    ----------
    session_set: SessionSet | str | list[int]
        The session set to get the sessions from. Can be a SessionSet enum, a string, or a list of session IDs.

    Returns
    -------
    list[int]
        The list of sessions from the session set.
    """
    if isinstance(session_set, list):
        sessions = session_set
        session_set = SessionSet.CUSTOM
        return sessions, session_set

    if isinstance(session_set, str):
        session_set = SessionSet(session_set.lower())

    match session_set:
        case SessionSet.ALL:
            from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
            cache = EcephysProjectCache.from_warehouse(manifest=paths.ECEPHYS_MANIFEST_FILE)
            sessions = cache.get_session_table().index.to_list()

        case SessionSet.TEST:
            with open(paths.TEST_SESSIONS_FILE, 'r') as f:
                test_sessions = json.load(f)
            sessions = test_sessions

        case SessionSet.SELECTED:
            import pandas as pd
            file = paths.OUTPUT_BASE_DIR / "session_selection.csv"
            sessions = pd.read_csv(file)['session_id'].to_list()

        case SessionSet.OPTOTAG:
            raise NotImplementedError("Optotag sessions are not implemented yet")

        case SessionSet.CUSTOM:
            raise ValueError("Custom sessions need to be provided as a list of session IDs")

    return sessions, session_set


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
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--session_set', type=str, default=session_set,
            help=f"The session set to process the sessions from. Available sets: {', '.join(available_session_sets)}."
        )
        self.add_parameter_to_parser()

    def add_parameter_to_parser(self):
        """Add parameters to the argument parser."""
        parser = self.parser
        for param, default in self.default_parameters.items():
            parameter = self.parameters.get(param)
            arg_type = parameter.type
            if arg_type is None:
                arg_type = type(default).__name__
            help = str(parameter)  # Get help text for the parameter

            match arg_type:
                case 'bool':
                    if default is False:
                        parser.add_argument(f'--{param}', action='store_true', default=False, help=help)
                    else:
                        parser.add_argument(f'--no-{param}', dest=param, action='store_false', default=True, help=help)
                case 'list':
                    parser.add_argument(
                        f'--{param}',
                        type=str,
                        nargs='+',
                        default=default,
                        help=f"{help} Provide list input as space-separated values."
                    )
                case _:
                    try:
                        parser.add_argument(f'--{param}', type=arg_type, default=default, help=help)
                    except Exception as e:
                        raise TypeError(f"Invalid argument type for parameter '{param}': '{arg_type}'") from e

    def parse_args(self) -> dict[str, Any]:
        """Parse the arguments and return the dictionary of arguments for the `process_sessions()` function."""
        args = self.parser.parse_args()
        parameters = {key: getattr(args, key) for key in self.default_parameters.keys()}
        arguments = dict(session_set=args.session_set, parameters=parameters)
        return arguments


def get_log_info(process_function: Callable) -> str:
    """Get the log file for the process function with format <script_name>_<timestamp>"""
    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    try:
        script_name = Path(inspect.getfile(process_function)).stem
    except TypeError:
        script_name = process_function.__name__
    return script_name, timestamp


def log_file_path(script_name: str, timestamp: str) -> Path:
    batch_log_dir = paths.BATCH_LOG_DIR
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    return batch_log_dir / f"{script_name}_{timestamp}.log"


def parameters_file_path(script_name: str, timestamp: str) -> Path:
    return paths.BATCH_LOG_DIR / f"{script_name}_parameters_{timestamp}.json"


def write_to_log(file):
    """Context manager to write output to the log file."""
    @contextlib.contextmanager
    def _log():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = file
            sys.stderr = file
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _log()


def process_sessions(
    process_function: Callable[[int, ...], Any],
    session_set: SessionSet | str | list[int],
    parameters: dict[str, Any]
):
    """Process the sessions from the session set.

    Parameters
    ----------
    process_function: Callable
        The process function to execute for each session. The first argument must be the session ID.
    session_set: SessionSet | str | list[int]
        The session set to process the sessions from.
    parameters: dict[str, Any]
        The dictionary of parameters to pass to the process function.

    Returns
    -------
    None
    """
    sessions, session_set = get_sessions(session_set)

    # Get log file and parameters file
    script_name, timestamp = get_log_info(process_function)
    log_file = log_file_path(script_name, timestamp)
    parameters_file = parameters_file_path(script_name, timestamp)

    # Write to log file
    with open(log_file, 'w', encoding='utf-8') as f:
        # Function and arguments 
        f.write(f"Process script: {script_name}\n")
        f.write(f"Process function: {process_function.__name__}\n")
        f.write(f"Parameters:\n")
        for key, value in parameters.items():
            f.write(f"  {key}: {value}\n")

        # Session set and session ids
        f.write(f"\nProcessing session set: {session_set.name}\n")
        f.write(f"Session IDs: \n")
        for session in sessions:
            f.write(f"  {session}\n")

        # Write parameters to file
        with open(parameters_file, 'w', encoding='utf-8') as pf:
            json.dump(parameters, pf, indent=4)

        # Start processing
        f.write(f"\n\nProcessing started at {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Main processing loop
        with write_to_log(f):
            for session in sessions:
                f.write(f"\nProcessing session: {session}\n")
                f.write("-" * 80 + "\n\n")
                try:
                    process_function(session, **parameters)
                except Exception as e:
                    print(f"\n[ERROR] Exception occurred while processing session {session}:\n", file=f)
                    traceback.print_exc(file=f)
                f.write("\n" + "-" * 80 + "\n\n")

        # End processing
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"\nProcessing completed at {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n")

