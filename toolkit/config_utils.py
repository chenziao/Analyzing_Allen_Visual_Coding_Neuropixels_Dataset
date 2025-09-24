"""
Configuration utilities for resolving path variables in JSON configs.
"""
import json
import re
from pathlib import Path
from typing import Any


class PathConfigResolver:
    """Utility class to resolve path variables in configuration dictionaries."""
    
    def __init__(
        self,
        config_file: str | Path = None,
        config_dict: dict = None,
        root_dir: str | Path = None
    ):
        """
        Initialize the resolver with either a config file or config dictionary.
        
        Args:
            config_file: Path to JSON configuration file
            config_dict: Configuration dictionary
            root_dir: Root directory for resolving relative paths (defaults to config file's directory or current directory)
        """
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            # Default root_dir to the config file's directory
            if root_dir is None:
                root_dir = Path(config_file).parent
        elif config_dict:
            self.config = config_dict.copy()
            # Default root_dir to current directory if not specified
            if root_dir is None:
                root_dir = Path.cwd()
        else:
            raise ValueError("Either config_file or config_dict must be provided")
            
        self.root_dir = Path(root_dir).resolve()
        self.resolved_config = None
    
    def resolve_paths(self) -> dict[str, Any]:
        """
        Resolve all path variables in the configuration.
        
        Returns:
            Dictionary with resolved paths
        """
        if self.resolved_config is not None:
            return self.resolved_config
            
        # Create a copy to work with
        resolved = self.config.copy()
        
        # Keep resolving until no more variables are found
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            changed = False
            
            for key, value in resolved.items():
                if isinstance(value, str) and '${' in value:
                    new_value = self._resolve_string(value, resolved)
                    if new_value != value:
                        resolved[key] = new_value
                        changed = True
            
            if not changed:
                break
            iteration += 1
        
        # Convert to absolute paths relative to root_dir
        for key, value in resolved.items():
            if isinstance(value, str) and ('_dir' in key or '_path' in key):
                path = Path(value)
                if not path.is_absolute():
                    path = self.root_dir / path
                resolved[key] = str(path.resolve())
        
        self.resolved_config = resolved
        return resolved
    
    def _resolve_string(self, text: str, config: dict[str, Any]) -> str:
        """
        Resolve variables in a string using the provided config.
        
        Args:
            text: String potentially containing variables like ${var_name}
            config: Configuration dictionary to resolve variables from
            
        Returns:
            String with variables resolved
        """
        # Find all variables in the format ${variable_name}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name in config:
                return str(config[var_name])
            else:
                # Return original if variable not found
                return match.group(0)
        
        return re.sub(pattern, replace_var, text)
    
    def get_path(self, key: str) -> Path:
        """
        Get a resolved path as a Path object.
        
        Args:
            key: Configuration key
            
        Returns:
            Path object
        """
        return Path(self.get_str(key))
    
    def get_str(self, key: str) -> str:
        """
        Get a resolved configuration value as string.
        
        Args:
            key: Configuration key
            
        Returns:
            Resolved string value
        """
        resolved = self.resolve_paths()
        if key not in resolved:
            raise KeyError(f"Key '{key}' not found in configuration")
        
        return resolved[key]


def load_config_with_resolved_paths(config_file: str | Path, root_dir: str | Path = None) -> dict[str, Any]:
    """
    Convenience function to load and resolve a configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        root_dir: Root directory for resolving relative paths (defaults to config file's directory)
        
    Returns:
        Dictionary with resolved paths
    """
    resolver = PathConfigResolver(config_file=config_file, root_dir=root_dir)
    return resolver.resolve_paths()


# Example usage
if __name__ == "__main__":
    # Example with the path_config.json
    try:
        config = load_config_with_resolved_paths("path_config.json")
        print("Resolved configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print("Configuration file not found")
