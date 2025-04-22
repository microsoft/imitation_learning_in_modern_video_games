from pathlib import Path


def get_file_paths_from_split_file(path, base_path=None, relative_path=True):
    """
    Get files from the given split file, relative to base_path if given.
    :param path: Path to split file.
    :param base_path: Base path to add to files.
    :param relative_path: Whether to given path is relative to the current directory.
    :return: List of files.
    """
    path = Path(__file__).parent / Path(path) if relative_path else Path(path)
    with path.open("r") as f:
        if base_path is None:
            return f.read().splitlines()
        else:
            return [f"{base_path}/{path}" for path in f.read().splitlines()]


def is_file_in_split(split_files, file):
    """
    Check if the given file is in the split of files.
    :param split_files: List of files in split
    :param file: File to check.
    :return: True if the file is in the split, False otherwise.
    """
    return file in split_files


def is_file_in_split_file(path, file):
    """
    Check if the given file is in the given split file.
    :param path: Path to split file.
    :param file: File to check.
    :return: True if the file is in the split file, False otherwise.
    """
    path = Path(__file__).parent / Path(path)
    return file in get_file_paths_from_split_file(path)


def get_files_not_in_split(path, files):
    """
    Get files that are not in the given split file.
    :param path: Path to split file.
    :param files: Files to check.
    :return: List of files that are not in the split file.
    """
    if not path.exists():
        return files
    else:
        return list(set(files).difference(get_file_paths_from_split_file(path)))
