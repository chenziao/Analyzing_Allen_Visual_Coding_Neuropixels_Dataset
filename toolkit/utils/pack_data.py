"""Data packing utilities for filtering and zipping files."""

import zipfile
from fnmatch import fnmatch
from pathlib import Path


def match_patterns(
    file_path: Path,
    accept_patterns: list[str],
    filter_patterns: list[str] = []
) -> bool:
    """Check if file matches accept patterns and doesn't match filter patterns.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to check
    accept_patterns : List[str]
        List of patterns for files to accept. If empty, all files are accepted.
    filter_patterns : List[str]
        List of patterns for files to reject. If empty, no files are filtered.
    
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
    name = file_path.name
    path_str = str(file_path)
    accept = True
    # Check if file matches accept patterns
    if accept_patterns:
        accept = any(fnmatch(name, p) or fnmatch(path_str, p) for p in accept_patterns)
    # Check if file matches filter patterns
    if filter_patterns:
        accept = accept and not any(fnmatch(name, p) or fnmatch(path_str, p) for p in filter_patterns)
    return accept


def filter_and_zip(
    source_dir: Path,
    zip_path: Path,
    accept_patterns: list[str] = [],
    filter_patterns: list[str] = []
) -> None:
    """Filter files from source_dir and add them directly to a zip file.
    
    Parameters
    ----------
    source_dir : Path
        Source directory to search for files
    zip_path : Path
        Path where the zip file will be created
    accept_patterns : List[str]
        List of patterns for files to accept
    filter_patterns : List[str]
        List of patterns for files to reject
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for source_path in source_dir.rglob("*"):
            if not source_path.is_file():
                continue

            if not match_patterns(source_path, accept_patterns, filter_patterns):
                continue

            # Get relative path from source_dir for archive
            arcname = source_path.relative_to(source_dir)
            zipf.write(source_path, arcname)
    print(f"Zip file created: {zip_path}")

