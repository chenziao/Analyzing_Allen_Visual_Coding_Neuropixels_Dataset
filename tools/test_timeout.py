import sys
import os
import time

import add_path
from toolkit.utils.timeout_handler import timeout_context, run_with_timeout

TIMEOUT = 5

def sleep_function():
    """Function that sleeps for 10 seconds total."""
    for i in range(5):
        print(f"Sleeping for {i * 2} seconds...")
        time.sleep(2)
    print("Sleep function completed successfully")


def test_signal_timeout():
    """Test signal-based timeout context manager."""
    print("=== Testing Signal-based Timeout ===")
    try:
        with timeout_context(TIMEOUT, "Sleep test"):
            sleep_function()
        print("Signal timeout test completed successfully")
    except Exception as e:
        print(f"Signal timeout test failed: {e}")


def test_threaded_timeout():
    """Test threaded timeout function."""
    print("\n=== Testing Threaded Timeout ===")
    try:
        result = run_with_timeout(sleep_function, TIMEOUT, "Sleep test")
        print("Threaded timeout test completed successfully")
    except Exception as e:
        print(f"Threaded timeout test failed: {e}")


if __name__ == "__main__":
    test_signal_timeout()
    test_threaded_timeout()
