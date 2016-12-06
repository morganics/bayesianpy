import os

def get_path_to_parent_dir(filename):
    return os.path.dirname(os.path.abspath(filename))