import json
from pathlib import Path


class ConfigAccessor:
    """
    Class to access nested config dict via string key of variable names separated by '.'.
    """
    def __init__(self, config_dict: dict | str | Path):
        if isinstance(config_dict, (str, Path)):
            with open(config_dict, 'r') as f:
                self._config = json.load(f)
            self._config_file = Path(config_dict)
        elif isinstance(config_dict, dict):
            self._config = config_dict
            self._config_file = None
        else:
            raise TypeError(f"Expected dict, str, or Path, got {type(config_dict)}")

    def __repr__(self):
        return f"ConfigAccessor({self._config_file if self._config_file else 'Dict'})"

    def get(self, *args):
        """
        Retrieve a value from the nested config dict using a '.'-separated key string.

        Parameters
        ----------
        key_str : str
            Dot-separated sequence of keys (e.g. 'key_1.key_2.key_3')
        default : any, optional
            If specified, value to return if key path does not exist.
            If not specified, raise KeyError if key path does not exist.

        Returns
        -------
        Value from the config dict or `default` if specified.
        """
        if len(args) == 1:
            key_str = args[0]
            raise_error = True
        elif len(args) == 2:
            key_str, default = args
            raise_error = False
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")

        keys = key_str.split('.')
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                if raise_error:
                    raise KeyError(f"Key '{key}' in '{key_str}' not found in config")
                else:
                    return default
        return current

    def __call__(self, key_str, default=None):
        """Retrieve a value from the nested config dict using a '.'-separated key string.
        Return default value when key path does not exist. Default value is None if not specified.
        See `get` for more details.
        """
        return self.get(key_str, default)

    def __getitem__(self, key_str):
        """Retrieve a value from the nested config dict using a '.'-separated key string.
        Raise KeyError if key path does not exist.
        See `get` for more details.
        """
        return self.get(key_str)
