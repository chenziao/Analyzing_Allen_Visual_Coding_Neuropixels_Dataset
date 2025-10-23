"""
Timeout handling utilities for network operations and long-running processes.

This module provides context managers and utility functions for implementing
timeouts in various environments, including SLURM clusters where signal-based
timeouts may not work reliably.
"""

import signal
import threading
import contextlib
from typing import Callable, Any


class TimeoutError(Exception):
    """Custom timeout exception for timeout operations."""
    pass


@contextlib.contextmanager
def timeout_context(timeout_seconds: int, operation_name: str = "operation"):
    """Context manager for timeout operations using signals.
    
    This context manager uses SIGALRM signals to interrupt long-running
    operations. Works well for network operations and system calls that
    can be interrupted by signals.
    
    Parameters
    ----------
    timeout_seconds : int
        Timeout duration in seconds. If 0, timeout is disabled.
    operation_name : str, default "operation"
        Name of the operation for error messages
        
    Yields
    ------
    None
        Control to the code block within the context
        
    Raises
    ------
    TimeoutError
        If the operation exceeds the timeout duration (when timeout_seconds > 0)
        
    Examples
    --------
    >>> with timeout_context(60, "Data download"):
    ...     data = download_large_file()
    >>> 
    >>> # Disable timeout
    >>> with timeout_context(0, "Data download"):
    ...     data = download_large_file()
    """
    if timeout_seconds <= 0:
        # No timeout - just yield control
        yield
        return
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        yield
    except TimeoutError as e:
        print(f"[TIMEOUT] {e}")
        raise
    finally:
        signal.alarm(0)  # Cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore original handler


def run_with_timeout(func: Callable[[], Any], timeout_seconds: int, operation_name: str = "operation") -> Any:
    """Run a function with timeout using threading (SLURM-friendly).
    
    This function runs the provided function in a separate thread and waits
    for completion or timeout. This approach works better in SLURM environments
    where signal-based timeouts may not be reliable.
    
    Parameters
    ----------
    func : callable
        Function to execute (should take no arguments)
    timeout_seconds : int
        Timeout duration in seconds. If 0, timeout is disabled.
    operation_name : str, default "operation"
        Name of the operation for error messages
        
    Returns
    -------
    Any
        Result of the function execution
        
    Raises
    ------
    TimeoutError
        If the function exceeds the timeout duration (when timeout_seconds > 0)
    Exception
        Any exception raised by the function
        
    Examples
    --------
    >>> def download_data():
    ...     return cache.get_session_data(session_id)
    >>> 
    >>> data = run_with_timeout(download_data, 300, "Session data download")
    >>> 
    >>> # Disable timeout
    >>> data = run_with_timeout(download_data, 0, "Session data download")
    """
    if timeout_seconds <= 0:
        # No timeout - just run the function directly
        return func()
    
    result = [None]
    exception = [None]
    completed = [False]
    
    def target():
        try:
            result[0] = func()
            completed[0] = True
        except Exception as e:
            exception[0] = e
            completed[0] = True
    
    # Start the function in a separate thread
    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running - timeout occurred
        print(f"[TIMEOUT] {operation_name} timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


