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
            if type(figname) is dict:
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
    default_params = {key: params_dict.get(key, value) for key, value in default_params.items()}
    if enter_parameters:
        print('Enter parameters:')
        for key, value in default_params.items():
            p = input(f'{key} (default: {value}) : ')
            if p != '':
                p = eval(p)
                default_params[key] = list(p) if type(p) is tuple else p
    params_dict.update(default_params)
    return tuple(default_params.values())
