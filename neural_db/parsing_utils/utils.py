import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

ATTACH_N_WORD_THRESHOLD = 20
MIN_WORDS_PER_CHUNK = 50
CHUNK_THRESHOLD = 150


def display_index(filename):
    display_df = pd.read_csv(filename)  # for display preserved casing
    display_df = display_df[
        ["id", "filename", "page", "passage", "display", "highlight"]
    ]  # select only relevant columns
    input_series = display_df["passage"]  # actual input sentence
    return display_df, input_series


def ensure_valid_encoding(text):
    return text.encode("utf-8", "replace").decode("utf-8")


def clean_text(text):
    return text.replace("\t", " ").replace(",", " ").replace("\n", " ").strip()


def chunk_text(text: str):
    sentences = sent_tokenize(text)
    if len(sentences) == 1:
        return [text]

    words_per_sentence = [len(word_tokenize(sent)) for sent in sentences]
    if sum(words_per_sentence) < CHUNK_THRESHOLD:
        return [text]

    chunks = []
    cur_word_count = 0
    start_idx = 0

    for idx in range(len(sentences)):
        word_count = words_per_sentence[idx]
        if cur_word_count < MIN_WORDS_PER_CHUNK:
            cur_word_count += word_count
        else:
            chunks.append(" ".join(sentences[start_idx:idx]))
            start_idx = idx
            cur_word_count = word_count

    if start_idx != len(sentences):
        final_chunk = " ".join(sentences[start_idx : len(sentences)])
        if len(chunks) > 0 and cur_word_count < MIN_WORDS_PER_CHUNK:
            chunks[-1] += final_chunk
        else:
            chunks.append(final_chunk)

    return chunks
