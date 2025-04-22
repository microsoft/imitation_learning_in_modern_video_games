from pathlib import Path

MINERL_VIDEO_FILE_SUFFIX = ".mp4"
MINERL_ACTION_FILE_SUFFIX = ".jsonl"
CSGO_FILE_SUFFIX = ".hdf5"
EMBEDDING_FILE_SUFFIX = ".npz"
EMBEDDINGS_FOLDER = "embeddings"


def is_action_file(actions_file_suffix, filename):
    return filename.endswith(actions_file_suffix)


def get_minerl_video_file_from_action_file(filename):
    # MineRL videos are same name but .jsonl is replaced with .mp4
    return filename.replace(MINERL_ACTION_FILE_SUFFIX, MINERL_VIDEO_FILE_SUFFIX)


def get_embedding_filename(encoder_family, encoder_name):
    encoder_family = encoder_family.lower().replace("/", "")
    encoder_name = encoder_name.lower().replace("/", "")
    return f"embed_{encoder_family}_{encoder_name}{EMBEDDING_FILE_SUFFIX}"


def get_pretrained_encoder_dirname(encoder_family, encoder_name):
    encoder_family = encoder_family.lower().replace("/", "")
    encoder_name = encoder_name.lower().replace("/", "")
    return Path(EMBEDDINGS_FOLDER) / encoder_family / encoder_name


def get_embedding_file(encoder_dirname, filename):
    filename = Path(filename)
    data_dir = filename.parent / encoder_dirname
    embedding_filename = filename.name.replace(filename.suffix, EMBEDDING_FILE_SUFFIX)
    embedding_path = data_dir / embedding_filename
    assert embedding_path.is_file(), f"Could not find file {embedding_path})"
    return embedding_path
