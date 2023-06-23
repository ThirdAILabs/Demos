import hashlib


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

