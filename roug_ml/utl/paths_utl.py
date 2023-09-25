import os


def create_dir_path(in_path):
    """
    Create directory path
    :param in_path: path to create the folder in
    :type in_path: str
    :return:
    :rtype: str
    """
    file_path = os.path.join(in_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


def create_dir(in_path: str, in_folder_name: str = "", in_exist_ok: bool = False) -> str:
    """
    Create directory from path and folder name
    :param in_path: path to create the folder in
    :param in_folder_name: subfolder name (default is empty, just create parent folder)
    :param in_exist_ok: Boolean (default is False). If set to True, the function won't raise an
    error if the directory already exists. If False and directory already exists, an OSError is
    raised.
    :return: The absolute path of the created directory.
    """
    file_path = os.path.join(in_path, in_folder_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=in_exist_ok)
    return file_path
