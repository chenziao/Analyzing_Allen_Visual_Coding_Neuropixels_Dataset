import importlib

from types import ModuleType
from typing import Any


def reload_module(module_name : str | ModuleType, *variable_names : str) -> ModuleType | Any | tuple[Any, ...]:
    """Reload a module in runtime.

    Parameters
    ----------
    module_name : str | ModuleType
        The name of the module or the module itself to reload.
    *variable_names : str
        The names of the variables to return.
        If no variable names are provided, the module itself is returned.

    Returns
    -------
    ModuleType | Any | tuple[Any, ...]
        The reloaded module or the variable.
    """
    if isinstance(module_name, str):
        module = importlib.import_module(module_name)
    else:
        module = module_name
    importlib.reload(module)
    if len(variable_names) > 0:
        vars = tuple(getattr(module, var) for var in variable_names)
        return vars[0] if len(vars) == 1 else vars
    else:
        return module

