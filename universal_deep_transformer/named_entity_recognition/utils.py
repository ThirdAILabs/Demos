from datasets import load_dataset
import json
import os
import pandas as pd


TAG_MAP = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

entries = list(TAG_MAP.keys())


def save_dataset_as_jsonl(filename, loaded_data):
    data = {"source": [], "target": []}
    for example in loaded_data:
        data["source"].append(" ".join(example["tokens"]))
        data["target"].append(" ".join([entries[tag] for tag in example["ner_tags"]]))

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def download_conll_dataset_as_file(subset):
    # Load dataset
    dataset = load_dataset("conll2003")
    loaded_data = dataset[f"{subset}"]
    filename = f"{subset}_ner_data.csv"
    if os.path.exists(filename):
        return filename

    save_dataset_as_jsonl(filename, loaded_data)
    return filename
