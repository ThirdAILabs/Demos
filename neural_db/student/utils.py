import hashlib
import math
import random


def clean_text(text):
    return text.encode("utf-8", "replace").decode("utf-8").lower()


def hash_file(path: str):
    """https://stackoverflow.com/questions/22058048/hashing-a-file-in-python"""
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()


def hash_string(string: str):
    sha1 = hashlib.sha1(bytes(string, "utf-8"))
    return sha1.hexdigest()


def random_sample(sequence, k):
    if len(sequence) > k:
        return random.sample(sequence, k)
    mult_factor = math.ceil(k / len(sequence))
    return (sequence * mult_factor)[:k]


def move_between_directories(src, dest):
    import os
    import shutil

    # gather all files
    allfiles = os.listdir(src)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(src, f)
        dst_path = os.path.join(dest, f)
        shutil.move(src_path, dst_path)
