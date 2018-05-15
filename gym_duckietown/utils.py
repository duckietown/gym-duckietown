import os

def get_file_path(sub_dir, file_name, default_ext):
    """
    Get the absolute path of a resource file, which may be relative to
    the gym_duckietown module directory, or an absolute path.

    This function is necessary because the simulator may be imported by
    other packages, and we need to be able to load resources no matter
    what the current working directory is.
    """

    assert '.' not in default_ext
    assert '/' not in default_ext

    # If this is already a real path
    if os.path.exists(file_name):
        return file_name

    # Get the directory this module is located in
    abs_path_module = os.path.realpath(__file__)
    module_dir, _ = os.path.split(abs_path_module)

    file_path = os.path.join(module_dir, sub_dir, file_name)

    if '.' not in file_name:
        file_path += '.' + default_ext

    return file_path
