import functools
import re
from typing import Callable, List

import fitz
import pandas as pd
from nltk.tokenize import sent_tokenize

from .utils import ensure_valid_encoding, chunk_text, ATTACH_N_WORD_THRESHOLD

# TODO: Remove senttokenize
# TODO: Limit paragraph length

def para_is_complete(para):
    endings = [".", "?", "!", '."', ".'"]
    return functools.reduce(
        lambda a, b: a or b,
        [para.endswith(end) for end in endings],
    )


# paragraph = {"page_no": [block_id,...], "pagen_no_2":[blicksids, ...]}
def process_pdf_file(filename):
    try:
        rows = []
        prev = ""
        prev_n_words = float("inf")
        doc = fitz.open(filename)
        paras = []
        for page_no, page in enumerate(doc):
            temp_page = page.get_text("blocks")
            for i, t in enumerate(temp_page):
                if t[-1] == 0:
                    temp = {}
                    temp[page_no] = [i]
                    current = sent_tokenize(
                        t[4].strip().replace("\r\n", " ").replace("\n", " ")
                    )
                    current = " ".join(current)

                    if (
                        len(paras) > 0
                        and prev != ""
                        and (
                            not para_is_complete(paras[-1][0])
                            or prev_n_words < ATTACH_N_WORD_THRESHOLD
                        )
                    ):
                        attach = True
                    else:
                        attach = False

                    if attach and len(paras) > 0:
                        temp_prev = paras[-1][3]
                        if page_no in temp_prev.keys():
                            temp_prev[page_no].extend(temp[page_no])
                        else:
                            temp_prev[page_no] = temp[page_no]
                        paras[-1] = (
                            paras[-1][0] + " " + current,  # text
                            paras[-1][1],  # page number
                            doc.name,  # pdf filename
                            temp_prev,  # page to block dictionary
                        )
                    else:
                        prev = current
                        paras.append((current, page_no, doc.name, temp))

                    # Occurrences of space is proxy for number of words.
                    # If there are 10 words or less, this paragraph is
                    # probably just a header.
                    prev_n_words = len(current.split(" "))

        paras = [
            (chunk, page_no, docname, temp)
            for passage, page_no, docname, temp in paras
            for chunk in chunk_text(passage)
        ]
        for para in paras:
            if len(para) > 0:
                sent = re.sub(
                    " +",
                    " ",
                    str(para[0])
                    .replace("\t", " ")
                    .replace(",", " ")
                    .replace("\n", " ")
                    .strip(),
                )
                if len(sent) > 0:
                    rows.append(
                        (sent.lower(), sent, para[1], para[2], str(para[3]))
                    )

        return rows, True
    except Exception as e:
        print(e.__str__())
        return "Cannot process pdf file:" + filename, False


def create_train_df(elements):
    df = pd.DataFrame(
        index=range(len(elements)),
        columns=["passage", "para", "filename", "page", "display", "highlight"],
    )
    for i, elem in enumerate(elements):
        sents = sent_tokenize(elem[1])
        sents = list(map(lambda x: x.lower(), sents))
        passage = " ".join(sents)
        # elem[-1] is id
        df.iloc[i] = [passage, passage, elem[3], elem[2], elem[1], elem[4]]
    for column in ["passage", "para", "display"]:
        df[column] = df[column].apply(ensure_valid_encoding)
    return df


def num_classes(filename):
    return len(pd.read_csv(filename, encoding="utf-8"))


def highlight_pdf(input_filename, output_filename, highlight):
    doc = fitz.open(input_filename)
    for key, val in highlight.items():
        page = doc[key]
        blocks = page.get_text("blocks")
        for i, b in enumerate(blocks):
            if i in val:
                rect = fitz.Rect(b[:4])
                page.add_highlight_annot(rect)
    doc.save(output_filename)
