import numpy as np
import time
import os
import pickle

import thirdai


def get_doc_starts_and_ends(doc_lens):
    doc_starts = [0] * len(doc_lens)
    for i in range(1, len(doc_lens)):
        doc_starts[i] = doc_starts[i - 1] + doc_lens[i - 1]
    return [(start, start + length) for (start, length) in zip(doc_starts, doc_lens)]


def create_index(input_path, save_path, hashes_per_table=7, num_tables=32):

    centroids = np.load(f"{input_path}/centroids.npy")
    all_centroid_ids = np.load(f"{input_path}/all_centroid_ids.npy")
    current_chunk = np.load(f"{input_path}/encodings0_float32.npy")

    with open(f"{input_path}/doclens.npy", "rb") as f:
        doc_offsets = get_doc_starts_and_ends(np.load(f))
    with open(f"{input_path}/collection.tsv") as f:
        texts = [line.split("\t")[1].strip() for line in f.readlines()]
    with open(f"{input_path}/collection.tsv") as f:
        ids = [line.split("\t")[0].strip() for line in f.readlines()]

    current_chunk_num = 0
    data_dim = current_chunk.shape[1]
    current_chunk_start = 0
    current_chunk_end = len(current_chunk)

    doc_index = thirdai.search.DocRetrieval(
        centroids=centroids,
        num_tables=num_tables,
        dense_input_dimension=data_dim,
        hashes_per_table=hashes_per_table,
    )

    current_start = time.time()
    original_start = time.time()
    for doc_id, (doc_start, doc_end) in enumerate(doc_offsets):
        if doc_id % 1000 == 0:
            diff = time.time() - current_start
            current_start = time.time()
            print(f"Adding last 100 docs took {diff}")
            print(
                "We have processed", doc_id, "docs out of", len(doc_offsets), flush=True
            )
            print(f"Time elapsed so far: {time.time() - original_start}")
        if doc_end > current_chunk_end:
            next_chunk = np.load(
                f"{input_path}/encodings{current_chunk_num + 1}_float32.npy"
            )
            next_chunk_start = current_chunk_end
            to_add = np.concatenate(
                (
                    current_chunk[doc_start - current_chunk_start :],
                    current_chunk[: doc_end - next_chunk_start],
                )
            )
            current_chunk = next_chunk
            current_chunk_start = next_chunk_start
            current_chunk_end = current_chunk_start + len(current_chunk)
            current_chunk_num += 1
        else:
            start = doc_start - current_chunk_start
            end = doc_end - current_chunk_start
            to_add = current_chunk[start:end]
        doc_index.add_doc(
            doc_id=ids[doc_id],
            doc_text=texts[doc_id],
            doc_embeddings=to_add,
            doc_centroid_ids=all_centroid_ids[doc_start:doc_end],
        )

    doc_index.serialize_to_file(save_path)


import argparse

parser = argparse.ArgumentParser(description="Build and save a document search index.")
parser.add_argument(
    "input_path",
    help="""
      The path to the directory containing all of the necessary files for 
      building. These files are: doclens.npy, a one dimensional numpy array 
      containing the number of embeddings in each document. embeddings_i.npy
      for i from 0 to an arbitrarily large number. These are 2D arrays that 
      contain the precomputed embeddings for each document. When concatenated 
      they should be all of the embeddings for all documents in order. 
      centroids.npy, a collection of calculated centroids of the embeddings.
      collection.npy, a tsv where the first column is the id of the ith document
      and the second column is the text of the ith document. 
      Finally, all_centroid_ids.npy, a one dimensional array 
      containing the id of the nearest centroid corresponding to each embedding. 
      """,
)
parser.add_argument(
    "save_path",
    help="""
        The file path to save the index to, e.g. /some/path/index.serialized. 
        Note that /some/path must exist.
      """,
)
args = parser.parse_args()

create_index(input_path=args.input_path, save_path=args.save_path)
