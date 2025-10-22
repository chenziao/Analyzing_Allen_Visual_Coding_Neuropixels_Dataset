import importlib
import sys

from types import ModuleType
from typing import Any


def reload_module(module_name : str | ModuleType, *variable_names : str) -> ModuleType | Any | tuple[Any, ...]:
    """Reload a module in runtime (useful for development and debugging).

    Parameters
    ----------
    module_name : str | ModuleType
        The name of the module or the module itself to reload.
    *variable_names : str
        The names of the variables to return.
        If the variable name is '*', all variables from the module are loaded to the current namespace.
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

    if len(variable_names) == 1 and variable_names[0] == '*':
        # load all variables from the module to the parent's namespace
        parent_frame = sys._getframe(1)
        exec(f'from {module.__name__} import *', parent_frame.f_globals, parent_frame.f_locals)
        return
    elif len(variable_names) > 0:
        vars = tuple(getattr(module, var) for var in variable_names)
        return vars[0] if len(vars) == 1 else vars
    else:
        return module
