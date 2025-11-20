"""Data packing utilities for filtering and zipping files."""

import zipfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Sequence


def match_patterns(
    file_path: Path | str,
    accept_patterns: Sequence[str] | str = [],
    filter_patterns: Sequence[str] | str = []
) -> bool:
    """Check if file matches accept patterns and doesn't match filter patterns.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to check
    accept_patterns : Sequence[str] | str
        Patterns to accept; if empty, all files accepted. All must match.
    filter_patterns : Sequence[str] | str
        Patterns to reject; if empty, none rejected. Any match rejects.
    
    Returns
    -------
    bool
        True if file should be included, False otherwise.
    
    Examples
    --------
    Examples of filter patterns:
    - "*power.nc" matches files ending with "power.nc" (e.g., "drifting_gratings_condition_beta_power.nc")
    - "csd.nc" matches any file named "csd.nc" in any subdirectory
    """
    file_path = Path(file_path)
    name = file_path.name
    path_str = str(file_path)
    accept = True
    # Check if file matches accept patterns
    if isinstance(accept_patterns, str):
        accept_patterns = [accept_patterns] if accept_patterns else []
    if accept_patterns:
        accept = all(fnmatch(name, p) or fnmatch(path_str, p) for p in accept_patterns)
    # Check if file matches filter patterns
    if isinstance(filter_patterns, str):
        filter_patterns = [filter_patterns] if filter_patterns else []
    if filter_patterns:
        accept = accept and not any(fnmatch(name, p) or fnmatch(path_str, p) for p in filter_patterns)
    return accept


def match_files(
    file_paths: Path | str | list[Path | str],
    accept_patterns: Sequence[str] | str = [],
    filter_patterns: Sequence[str] | str = []
) -> list[Path]:
    """Return list of files matching accept and filter patterns.
    
    Parameters
    ----------
    file_paths : Path | str | list[Path | str]
        Directory to search for files, or list of paths to files
    accept_patterns : Sequence[str] | str
        Patterns to accept; if empty, all files accepted. All must match.
    filter_patterns : Sequence[str] | str
        Patterns to reject; if empty, none rejected. Any match rejects.

    Returns
    -------
    list[Path]
        List of files matching the patterns
    """
    if isinstance(file_paths, Path | str):
        file_paths = Path(file_paths).rglob("*")
    matched_files = []
    for file_path in file_paths:
        if file_path.is_file() and match_patterns(file_path, accept_patterns, filter_patterns):
            matched_files.append(file_path)
    return matched_files


def filter_and_zip(
    source_dir: Path | str,
    zip_path: Path,
    accept_patterns: Sequence[str] | str = [],
    filter_patterns: Sequence[str] | str = []
) -> None:
    """Filter files from source_dir and add them directly to a zip file.
    
    Parameters
    ----------
    source_dir : Path | str
        Source directory to search for files
    zip_path : Path
        Path where the zip file will be created
    accept_patterns : Sequence[str] | str
        List of patterns for files to accept
    filter_patterns : Sequence[str] | str
        List of patterns for files to reject
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for source_path in match_files(source_dir, accept_patterns, filter_patterns):
            # Get relative path from source_dir for archive
            arcname = source_path.relative_to(source_dir)
            zipf.write(source_path, arcname)
    print(f"Zip file created: {zip_path}")

