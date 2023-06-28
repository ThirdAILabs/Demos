import hashlib
import os
from pathlib import Path

import requests


def chunks(path: Path):
    def get_name(dir_entry: os.DirEntry):
        return Path(dir_entry.path).name

    if path.is_dir():
        for entry in sorted(os.scandir(path), key=get_name):
            yield bytes(Path(entry.path).name, "utf-8")
            for chunk in chunks(Path(entry.path)):
                yield chunk
    elif path.is_file():
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                yield chunk


def hash_path(path: Path):
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    if not path.exists():
        raise ValueError("Cannot hash an invalid path.")
    for chunk in chunks(path):
        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_directory_size(directory: Path):
    size = 0
    for root, dirs, files in os.walk(directory):
        for name in files:
            size += os.stat(Path(root) / name).st_size
    return size


def http_get_with_error(*args, **kwargs):
    """Makes an HTTP GET request and raises an error if status code is not
    200.
    """
    response = requests.get(*args, **kwargs)
    if response.status_code != 200:
        raise FileNotFoundError(f"{response.status_code} error: {response.reason}")
    return response

def streamed_download(source, destination, on_progress):
    """Makes an HTTP GET request and raises an error if status code is not
    200.
    """
    # Streaming, so we can iterate over the response.
    response = requests.get(source, allow_redirects=True, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    size_so_far = 0
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            size_so_far += len(data)
            on_progress(fraction=size_so_far / total_size_in_bytes)
            file.write(data)
    
    if size_so_far != total_size_in_bytes:
        raise ValueError("Failed to download.")