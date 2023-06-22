import hashlib
import random
import math

def clean_text(text):
    return text.encode('utf-8', 'replace').decode('utf-8').lower()

def hash_file(path: str):
    """https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    """
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()

def random_sample(sequence, k):
    if len(sequence) > k:
        return random.sample(sequence, k)
    mult_factor = math.ceil(k / len(sequence))
    return (sequence * mult_factor)[:k]