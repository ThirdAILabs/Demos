from thirdai import bolt, dataset
import numpy as np
import csv
import mmh3
import re

def tokenize_to_svm(filename, output_dim, target_location=None, train=True, seed=42):
    if filename.find(".csv") == -1:
        raise ValueError("Only .csv files are supported")

    post = "_train" if train else "_test"
    if target_location is None:
        target_location = "preprocessed_data"
    target_location = target_location + post + ".svm"

    fw = open(target_location, "w")
    csvreader = csv.reader(open(filename, "r"))

    for line in csvreader:
        label = 1 if line[0] == "pos" else 0
        fw.write(str(label) + " ")

        sentence = re.sub(r"[^\w\s]", "", line[1])
        sentence = sentence.lower()
        ### BOLT TOKENIZER START
        tup = dataset.bolt_tokenizer(sentence, seed=seed, dim=output_dim)
        for idx, val in zip(tup[0], tup[1]):
            fw.write(str(idx) + ":" + str(val) + " ")
        ### BOLT TOKENIZER END

        fw.write("\n")
    fw.close()

    return target_location