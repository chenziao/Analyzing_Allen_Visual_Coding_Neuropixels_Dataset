from pathlib import Path
from toolkit.config_utils import PathConfigResolver

# Load path configuration
PATH_CONFIG_FILE = Path(__file__).parent / 'path_config.json'
resolver = PathConfigResolver(config_file=PATH_CONFIG_FILE)


# Cache directories
CACHE_BASE_DIR = resolver.get_path('cache_base_dir')
ECEPHYS_CACHE_DIR = resolver.get_path('ecephys_cache_dir')
ECEPHYS_MANIFEST_FILE = ECEPHYS_CACHE_DIR / "manifest.json"
REFERENCE_SPACE_CACHE_DIR = resolver.get_path('reference_space_cache_dir')
PROCESSED_DATA_CACHE_DIR = resolver.get_path('processed_data_cache_dir')

# Output directories
OUTPUT_BASE_DIR = resolver.get_path('output_base_dir')
RESULTS_DIR = resolver.get_path('results_dir')
FIGURE_DIR = resolver.get_path('figure_dir')
