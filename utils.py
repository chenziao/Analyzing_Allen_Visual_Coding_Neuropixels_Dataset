import os
import matplotlib.pyplot as plt

"""
Functions for pipeline
"""

def figure_display_function(config, session_id=None, ecephys_structure_acronym=None, savefig_kwargs={}):
    """Return a display function for figures
    If `savefig` is false in config, the returned function only call pyplot.show()
    If `savefig` is true, the returned function saves figure(s) in directory `figure_dir` in config.
    Figure format is determined by the extension name in `figure_format` in config.
        function argument `figname`:
            if a string, save the current figure with file name `figname.{format extension name}`.
            if a dictionary, save figures by key-value pairs {`figname`: figure handle}
    Other arguments:
        session_id, ecephys_structure_acronym: if not specified, use `analysis_object` in config.
        savefig_kwargs: other keyword arguments for pyplot.savefig()
    """
    if config['save_figure']:
        figure_format = config['figure_format']
        if figure_format[0] != '.':
            figure_format = '.' + figure_format
        session_id = session_id or config['analysis_object']['session_id']
        ecephys_structure_acronym = ecephys_structure_acronym or config['analysis_object']['ecephys_structure_acronym']
        figure_dir = os.path.join(config['figure_dir'], f'session_{session_id:d}_{ecephys_structure_acronym:s}')
        if not os.path.isdir(figure_dir):
            os.makedirs(figure_dir)
        kwargs = config['savefig_kwargs']
        kwargs.update(savefig_kwargs)
        def disp_func(figname):
            if isinstance(figname, dict):
                for fname, fig in figname.items():
                    fig.savefig(os.path.join(figure_dir, fname + figure_format), **kwargs)
            else:
                plt.savefig(os.path.join(figure_dir, figname + figure_format), **kwargs)
            plt.show()
    else:
        def disp_func(*args, **kwargs):
            plt.show()
    return disp_func

def get_parameters(default_params, params_dict, enter_parameters=False):
    """Enter parameters and overwrite the parameter dictionary.
    default_params: a dictionary of default parameter {parameter: value}.
    params_dict: parameter dictionary {parameter: value} to store selected parameters.
        If a parameter already exists in `params_dict`, it will be used as default.
    enter_parameters: If true, enter parameter value. Default value is used if nothing entered.
        If false, return value from the `params_dict` if exists, otherwise from `default_params`.
    Return: tuple of selected values for parameters in `default_params`.
    """
    default_params = {key: params_dict.get(key, value) for key, value in default_params.items()}
    if enter_parameters:
        print('Enter parameters:')
        for key, value in default_params.items():
            if isinstance(value, str):
                value = f'"{value}"'
            p = input(f'{key} (default: {value}) : ')
            if p != '':
                p = eval(p)
                default_params[key] = list(p) if type(p) is tuple else p
    params_dict.update(default_params)
    return tuple(default_params.values())

def redo_condition(enter_parameters=True):
    """Return a function that determines whether or not to reselect parameters and redo a process"""
    if enter_parameters:
        def whether_redo(*args, **kwargs):
            """Enter decision"""
            s = input('Continue with the selected parameter [y/n]?')
            return s and s[0].lower() == 'n'
    else:
        def whether_redo(*args, **kwargs):
            """Always skip redo"""
            return False
    return whether_redo
