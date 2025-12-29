"""
Test script for batch processing and logging.

This script does nothing but print messages to test the logging system.
It can randomly raise errors to test aborted session handling.
"""

import add_path
import random
import time

PARAMETERS = dict(
    sleep_time = dict(
        default = 1.0,
        type = float,
        help = "Time to sleep in seconds to simulate processing."
    ),
    error_rate = dict(
        default = 0.2,
        type = float,
        help = "Probability of raising an error (0.0 to 1.0)."
    ),
    message = dict(
        default = "Hello from test script",
        type = str,
        help = "Message to print during processing."
    ),
    verbose = dict(
        default = True,
        type = bool,
        help = "Print verbose output."
    )
)


def test_process(
    session_id: int,
    sleep_time: float = 1.0,
    error_rate: float = 0.2,
    message: str = "Hello from test script",
    verbose: bool = True
) -> None:
    """Test process function that simulates work and may raise errors."""
    
    print(f"Processing session {session_id}...")
    print(f"  Message: {message}")

    if verbose:
        print(f"  Sleep time: {sleep_time}s")
        print(f"  Error rate: {error_rate}")
    
    # Simulate some work
    time.sleep(sleep_time)
    
    # Randomly raise an error based on error_rate
    if random.random() < error_rate:
        raise RuntimeError(f"Simulated error for session {session_id}")
    
    print(f"  Session {session_id} completed successfully!")


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(test_process, **args)

