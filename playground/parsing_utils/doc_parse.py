from typing import List, Callable
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from docx import Document
import os
import parsing_utils.utils as utils

def get_elements(filename):
    temp = []
    document = Document(filename)
    prev_short = False
    for p in document.paragraphs:
        if len(p.text.strip()) > 3:
            if prev_short:
                temp[-1] = (temp[-1][0] + " " + p.text.strip(), filename)
            else:
                temp.append((p.text.strip(), filename))
            prev_short = len(word_tokenize(p.text.strip())) < utils.ATTACH_N_WORD_THRESHOLD
    temp = [
        (chunk, filename) 
        for passage, filename in temp
        for chunk in utils.chunk_text(passage)
    ]
    return temp, True

def create_train_df(elements, id_col):
    count = 0
    df = pd.DataFrame(index=range(len(elements)), columns=[id_col, "passage", "para", "filename", "page", "display"])
    for i, elem in enumerate(elements):
        sents = sent_tokenize(str(elem[0]))
        sents = [utils.clean_text(sent).lower() for sent in sents]
        passage = " ".join(sents).replace("\n"," ").strip()
        # elem[-1] is id
        df.iloc[i] = [elem[-1], passage, passage, elem[1], "0", str(elem[0].replace("\n"," "))]
        count = count + 1
    for column in ["passage", "para", "display"]:
        df[column] = df[column].apply(utils.ensure_valid_encoding)
    return df

def show_docx(PLATFORM, item):
    if PLATFORM == "Windows" or PLATFORM == "win32":
        os.startfile(item)
    else:
        os.system("open \"" + item + "\"")