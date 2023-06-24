import copy
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from student import utils
from student.qa import ContextArgs


class DocumentConfig:
    def __init__(
        self,
        id_col: str,
        n_new_ids: str,
        unsupervised_dataset: str,
        introduction_dataset: str,
        strong_cols: List[str],
        weak_cols: List[str],
        hash: str,
    ):
        self.id_col = id_col
        self.n_new_ids = n_new_ids
        self.unsupervised_dataset = unsupervised_dataset
        self.introduction_dataset = introduction_dataset
        self.strong_cols = strong_cols
        self.weak_cols = weak_cols
        self.hash = hash


# Three requirements
# consecutive ID range for each source
# consecutive ID range for new sources of each document object
# consecutive ID range in [0, total_n_ids) for doclist


class Reference:
    def __init__(self, id: int, text: str, source: str, page: Optional[int] = None):
        self._id = id
        self._text = text
        self._source = source
        self._page = page

    def id(self):
        return self._id

    def text(self):
        return self._text

    def source(self):
        return self._source

    def page(self):
        return self._page


class Documents:
    def list_new_sources(self) -> List[str]:
        raise NotImplementedError()

    def get_source_hash_to_id_offset_map(self):
        raise NotImplementedError()

    def get_config(self) -> DocumentConfig:
        raise NotImplementedError()

    def get_first_new_id(self) -> int:
        raise NotImplementedError()

    def get_num_new_ids(self) -> int:
        raise NotImplementedError()

    def get_reference(self, id: int) -> Reference:
        raise NotImplementedError()

    def get_context(self, id: int, context_args: ContextArgs) -> str:
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()


class CSV(Documents):
    def __init__(self, path, id_col, strong_cols, weak_cols, display_cols, id_offset=0):
        if id_offset > 0:
            raise ValueError("CSV does not support id_offset.")

        self.path = path
        self.df = pd.read_csv(self.path)
        for col in strong_cols + weak_cols + display_cols:
            self.df[col] = self.df[col].fillna("")

        self.display = self.df.apply(lambda row: "\n".join(row[display_cols]), axis=1)
        self.references = self.df.apply(
            lambda row: "\n".join(row[strong_cols + weak_cols]), axis=1
        )
        self.sources = pd.Series(
            [f"Source: {Path(self.path).name}"] * len(self.display)
        )
        self.doc_config = DocumentConfig(
            id_col=id_col,
            n_new_ids=len(self.df),
            unsupervised_dataset=path,
            introduction_dataset=path,
            strong_cols=strong_cols,
            weak_cols=weak_cols,
            hash=utils.hash_file(self.path),
        )

    def list_new_sources(self) -> List[str]:
        return [self.path]

    def get_source_hash_to_id_offset_map(self):
        return {self.doc_config.hash: 0}

    def get_config(self) -> DocumentConfig:
        return self.doc_config

    def get_first_new_id(self) -> int:
        return 0

    def get_num_new_ids(self) -> int:
        return self.doc_config.n_new_ids

    def get_reference(self, id: int) -> Reference:
        meta_row = self.df.iloc[id]
        display = self.display.iloc[id]
        return Reference(
            id=meta_row[self.doc_config.id_col],
            text=display,
            source=self.path,
            page=None,
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


# TODO(Geordie): instead of writing new CSV files, we can implement a new
# DataSource object. We can also do away with DocConfigs. We can add a
# nextLine method to each document, which produces a single row containing only
# a strong column and a weak column and nothing else.


class DocList:
    def __init__(self):
        self.docs: List[Documents] = []

    def list_sources(self) -> List[str]:
        return [src for doc in self.docs for src in doc.list_new_sources()]

    def get_source_hash_to_id_offset_map(self):
        return {
            fhash: id_offset
            for doc in self.docs
            for fhash, id_offset in doc.get_source_hash_to_id_offset_map().items()
        }

    def add_document(self, document: Documents):
        if len(self.docs) > 0:
            new_doc_id_col = document.get_config().id_col
            last_doc_id_col = self.docs[-1].get_config().id_col
            if new_doc_id_col != last_doc_id_col:
                raise ValueError(
                    "Tried to add document with mismatched id column to doc list."
                )

        if document.get_first_new_id() != self.get_num_ids():
            raise ValueError(
                "Tried to add document with invalid id offset to doc list."
            )

        self.docs.append(document)

    def clear(self):
        self.docs = []

    def get_num_ids(self):
        return sum([doc.get_num_new_ids() for doc in self.docs])

    def _get_doc_for_id(self, id: int) -> Documents:
        # Iterate through docs in reverse order
        for i in range(len(self.docs) - 1, -1, -1):
            if self.docs[i].get_first_new_id() <= id:
                return self.docs[i]
        raise ValueError(f"Unable to get document for id {id}.")

    def get_reference(self, id: int) -> Reference:
        return self._get_doc_for_id(id).get_reference(id)

    def get_context(self, id: int, context_args: ContextArgs) -> str:
        return self._get_doc_for_id(id).get_context(id, context_args)

    def save_meta(self, directory: Path):
        for i, doc in enumerate(self.docs):
            subdir = directory / str(i)
            os.mkdir(subdir)
            doc.save_meta(subdir)

    def load_meta(self, directory: Path):
        for i, doc in enumerate(self.docs):
            subdir = directory / str(i)
            doc.load_meta(subdir)
