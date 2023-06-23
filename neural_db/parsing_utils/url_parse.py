from collections import deque
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from trafilatura import extract
from trafilatura.settings import use_config
from url_normalize import url_normalize

from .utils import ensure_valid_encoding

# Set headers of request to mimic a browser to retrieve the same html rendered in a browser
# https://stackoverflow.com/questions/27652543/how-to-use-python-requests-to-fake-a-browser-visit-a-k-a-and-generate-user-agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


def get_all_urls(base_url, max_crawl_depth):
    crawled_urls = set()
    valid_urls = set()
    queue = deque([(base_url, 0)])
    crawled_urls.add(base_url)
    print(f"crawl depth: {max_crawl_depth}", flush=True)

    while queue:
        url, depth = queue.popleft()

        if depth > max_crawl_depth:
            continue

        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                print(f"Skipping {url} (status code {response.status_code})", flush=True)
                continue
            else:
                print(f"Adding {url}", flush=True)
                valid_urls.add((url, response))
        except Exception as error:
            print(f"Skipping {url} (with error {error})", flush=True)
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            next_url = url_normalize(urljoin(base_url, href))

            url_exact_match = True
            if next_url not in crawled_urls and (
                next_url[: len(base_url)] == base_url or not url_exact_match
            ):
                crawled_urls.add(next_url)
                queue.append((next_url, depth + 1))

    print(f"Total links found: {len(valid_urls)}", flush=True)
    return valid_urls


def process_url(url, response):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        length_function=len,
    )
    elem = []

    if not response:
        try:
            response = requests.get(url, headers=HEADERS)
        except:
            return f"cannot extract text from {url}", False
        
    if response.status_code != 200:
        return f"cannot extract text from {url}", False

    index = response.text
    result = extract(
        index,
        include_formatting=False,
        include_comments=False,
        include_links=False,
        include_tables=False,
        favor_precision=True,
        config=config,
    )
    if not result:
        return f"cannot extract text from {url}", False

    texts = text_splitter.split_text(result)

    for text in texts:
        text = str(text).strip().replace("\r\n", " ").replace("\n", " ")
        row = [text, url]
        elem.append(row)

    return elem, True


def create_train_df(elements):
    df = pd.DataFrame(
        index=range(len(elements)),
        columns=["text", "display", "url"],
    )
    for i, elem in enumerate(elements):
        sents = sent_tokenize(elem[0])
        sents = list(map(lambda x: x.lower(), sents))
        text = " ".join(sents)
        df.iloc[i] = [
            text,
            elem[0],
            elem[1],
        ]
    for column in ["text", "display"]:
        df[column] = df[column].apply(ensure_valid_encoding)
    return df
