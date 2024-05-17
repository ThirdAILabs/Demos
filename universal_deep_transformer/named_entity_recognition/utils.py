from datasets import load_dataset
import json
import os


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
    with open(filename, "w") as file:
        for example in loaded_data:
            data = {
                "source": example["tokens"],
                "target": [entries[tag] for tag in example["ner_tags"]],
            }
            file.write(json.dumps(data) + "\n")


def download_conll_dataset_as_file(subset):
    # Load dataset
    dataset = load_dataset("conll2003", trust_remote_code=True)
    loaded_data = dataset[f"{subset}"]
    filename = f"{subset}_ner_data.jsonl"
    if os.path.exists(filename):
        return filename

    save_dataset_as_jsonl(filename, loaded_data)
    return filename
