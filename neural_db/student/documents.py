import copy
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from parsing_utils import doc_parse, pdf_parse, url_parse
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


# Base class for PDF and DOCX classes because they share the same logic.
class Extracted(Documents):
    def __init__(
        self,
        files: List[str],
        expected_id_col: str,
        hash_to_id_offset={},
        next_id_offset: int = 0,
        on_extract_success: Callable = lambda **kwargs: None,
        on_extract_failed: Callable = lambda **kwargs: None,
        on_extraction_complete: Callable = lambda **kwargs: None,
    ):
        self.first_new_id = next_id_offset

        if len(files) == 0:
            raise ValueError("Received empty files list.")

        self.source_files = files
        train_file = f"train.csv"
        intro_file = f"intro.csv"

        (
            self.new_samples_df,
            self.all_samples_df,
            self.hash_to_id_offset,
        ) = self.process_data(
            files=files,
            id_col=expected_id_col,
            hash_to_id_offset=hash_to_id_offset,
            next_id_offset=next_id_offset,
            on_extract_success=on_extract_success,
            on_extract_failed=on_extract_failed,
            on_extraction_complete=on_extraction_complete,
        )

        self.new_samples_df.index = self.new_samples_df[expected_id_col]
        self.new_samples_df.to_csv(intro_file, index=False)
        self.all_samples_df.to_csv(train_file, index=False)

        self.doc_config = DocumentConfig(
            id_col=expected_id_col,
            n_new_ids=len(self.new_samples_df),
            unsupervised_dataset=train_file,
            introduction_dataset=intro_file,
            strong_cols=["passage"],
            weak_cols=["para"],
            hash=utils.hash_file(train_file),
        )

    def process_data(
        self,
        files: List[str],
        id_col: str,
        hash_to_id_offset: dict,
        next_id_offset: int,
        on_extract_success: Callable,
        on_extract_failed: Callable,
        on_extraction_complete: Callable,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def list_new_sources(self) -> List[str]:
        return [Path(src).name for src in self.new_samples_df["filename"].unique()]

    def get_source_hash_to_id_offset_map(self):
        return self.hash_to_id_offset

    def get_config(self) -> DocumentConfig:
        return self.doc_config

    def get_first_new_id(self) -> int:
        return self.first_new_id

    def get_num_new_ids(self) -> int:
        return self.doc_config.n_new_ids

    def _get_row_with_id(self, id: int) -> int:
        first_id = self.get_first_new_id()
        return self.new_samples_df.iloc[id - first_id]

    def get_reference(self, id: int) -> Reference:
        row = self._get_row_with_id(id)
        return Reference(
            id=row[self.doc_config.id_col],
            text=row["display"],
            source=row["filename"],
            page=int(row["page"]),
        )

    def get_context(self, doc_id, context_args) -> str:
        if doc_id not in self.new_samples_df[self.doc_config.id_col]:
            raise ("Document id not in document.")

        if hasattr(context_args, "chunk_radius"):
            chunk_radius = context_args.chunk_radius
            url = self.new_samples_df.loc[
                self.new_samples_df[self.doc_config.id_col] == doc_id
            ]["filename"].iat[0]
            source_entries = self.new_samples_df.loc[
                self.new_samples_df["filename"] == url
            ]
            chunks = source_entries["passage"].tolist()

            center_chunk_idx = doc_id - self.first_new_id

            window_start = center_chunk_idx - chunk_radius // 2
            window_end = window_start + chunk_radius

            if window_start < 0:
                window_start = 0
                window_end = min(chunk_radius, len(chunks))
            elif window_end > len(chunks):
                window_end = len(chunks)
                window_start = max(0, len(chunks) - chunk_radius)

            window_chunks = chunks[window_start:window_end]
            return "\n".join(window_chunks)

        return ""

    def save_meta(self, directory: Path):
        # Save source files
        for file in self.source_files:
            shutil.copy(file, directory)

    def load_meta(self, directory: Path):
        # Update file paths to use metadata directory since checkpoints can be
        # used in a different location from where they are first created.
        self.source_files = [
            str(directory / Path(file).name) for file in self.source_files
        ]
        unsupervised_dataset_name = Path(self.doc_config.unsupervised_dataset).name
        introduction_dataset_name = Path(self.doc_config.introduction_dataset).name
        self.doc_config.unsupervised_dataset = str(
            directory / unsupervised_dataset_name
        )
        self.doc_config.introduction_dataset = str(
            directory / introduction_dataset_name
        )
        self.new_samples_df.to_csv(self.doc_config.introduction_dataset)
        self.all_samples_df.to_csv(self.doc_config.unsupervised_dataset)


class PDF(Extracted):
    def __init__(
        self,
        files: List[str],
        expected_id_col: str,
        hash_to_id_offset={},
        next_id_offset: int = 0,
        on_extract_success: Callable = lambda **kwargs: None,
        on_extract_failed: Callable = lambda **kwargs: None,
        on_extraction_complete: Callable = lambda **kwargs: None,
    ):
        super().__init__(
            files=files,
            expected_id_col=expected_id_col,
            hash_to_id_offset=hash_to_id_offset,
            next_id_offset=next_id_offset,
            on_extract_success=on_extract_success,
            on_extract_failed=on_extract_failed,
            on_extraction_complete=on_extraction_complete,
        )

    def process_data(
        self,
        files: List[str],
        id_col: str,
        hash_to_id_offset: dict,
        next_id_offset: int,
        on_extract_success: Callable,
        on_extract_failed: Callable,
        on_extraction_complete: Callable,
    ) -> pd.DataFrame:
        # Extract elements from each file
        bad_files = []
        all_elements = []
        new_elements = []
        hash_to_id_offset = copy.deepcopy(hash_to_id_offset)

        for f in files:
            elem, success = pdf_parse.process_pdf_file(f)

            if success:
                on_extract_success(file=f)
                fhash = utils.hash_file(f)
                if fhash in hash_to_id_offset:
                    id_offset = hash_to_id_offset[fhash]
                else:
                    id_offset = next_id_offset
                    hash_to_id_offset[fhash] = id_offset
                    next_id_offset += len(elem)
                    new_elements.extend(
                        [
                            (*el, i)
                            for el, i in zip(
                                elem, range(id_offset, id_offset + len(elem))
                            )
                        ]
                    )
                all_elements.extend(
                    [
                        (*el, i)
                        for el, i in zip(elem, range(id_offset, id_offset + len(elem)))
                    ]
                )

            else:
                on_extract_failed(file=f)
                bad_files.append(f)

        if len(all_elements) == 0:
            raise RuntimeError("Unable to extract data from files.")

        on_extraction_complete(elements=all_elements, bad_files=bad_files)
        all_samples_df = pdf_parse.create_train_df(all_elements, id_col=id_col)
        new_samples_df = pdf_parse.create_train_df(new_elements, id_col=id_col)

        return new_samples_df, all_samples_df, hash_to_id_offset


class DOCX(Extracted):
    def __init__(
        self,
        files: List[str],
        expected_id_col: str,
        hash_to_id_offset={},
        next_id_offset: int = 0,
        on_extract_success: Callable = lambda **kwargs: None,
        on_extract_failed: Callable = lambda **kwargs: None,
        on_extraction_complete: Callable = lambda **kwargs: None,
    ):
        super().__init__(
            files=files,
            expected_id_col=expected_id_col,
            hash_to_id_offset=hash_to_id_offset,
            next_id_offset=next_id_offset,
            on_extract_success=on_extract_success,
            on_extract_failed=on_extract_failed,
            on_extraction_complete=on_extraction_complete,
        )

    def process_data(
        self,
        files: List[str],
        id_col: str,
        hash_to_id_offset: dict,
        next_id_offset: int,
        on_extract_success: Callable,
        on_extract_failed: Callable,
        on_extraction_complete: Callable,
    ) -> pd.DataFrame:
        # Extract elements from each file
        bad_files = []
        all_elements = []
        new_elements = []
        hash_to_id_offset = copy.deepcopy(hash_to_id_offset)

        for f in files:
            elem, success = doc_parse.get_elements(f)
            if success:
                on_extract_success(file=f)
                fhash = utils.hash_file(f)
                if fhash in hash_to_id_offset:
                    id_offset = hash_to_id_offset[fhash]
                else:
                    id_offset = next_id_offset
                    hash_to_id_offset[fhash] = id_offset
                    next_id_offset += len(elem)
                    new_elements.extend(
                        [
                            (*el, i)
                            for el, i in zip(
                                elem, range(id_offset, id_offset + len(elem))
                            )
                        ]
                    )
                all_elements.extend(
                    [
                        (*el, i)
                        for el, i in zip(elem, range(id_offset, id_offset + len(elem)))
                    ]
                )

            else:
                on_extract_failed(file=f)
                bad_files.append(f)

        if len(all_elements) == 0:
            raise RuntimeError("Unable to extract data from files.")

        on_extraction_complete(elements=all_elements, bad_files=bad_files)
        all_samples_df = doc_parse.create_train_df(all_elements, id_col=id_col)
        new_samples_df = doc_parse.create_train_df(new_elements, id_col=id_col)

        return new_samples_df, all_samples_df, hash_to_id_offset


class URL(Documents):
    def __init__(
        self,
        base_url: str,
        expected_id_col: str,
        hash_to_id_offset={},
        next_id_offset: int = 0,
        scrape_depth: int = 0,
    ):
        self.base_url = base_url
        self.scrape_depth = scrape_depth
        self.first_new_id = next_id_offset

        train_file = f"train.csv"
        intro_file = f"intro.csv"

        (
            self.new_samples_df,
            self.all_samples_df,
            self.hash_to_id_offset,
        ) = self.process_data(
            base_url=base_url,
            id_col=expected_id_col,
            hash_to_id_offset=hash_to_id_offset,
            next_id_offset=next_id_offset,
            scrape_depth=scrape_depth,
        )
        self.new_samples_df.index = self.new_samples_df[expected_id_col]
        self.new_samples_df.to_csv(intro_file, index=False)
        self.all_samples_df.to_csv(train_file, index=False)

        self.doc_config = DocumentConfig(
            id_col=expected_id_col,
            n_new_ids=len(self.new_samples_df),
            unsupervised_dataset=train_file,
            introduction_dataset=intro_file,
            strong_cols=["text"],
            weak_cols=["text"],
            hash=utils.hash_file(train_file),
        )

    def process_data(
        self,
        base_url,
        id_col: str,
        hash_to_id_offset: dict,
        next_id_offset: int,
        scrape_depth: int,
    ) -> pd.DataFrame:
        # Extract elements from each file
        all_elements = []
        new_elements = []
        hash_to_id_offset = copy.deepcopy(hash_to_id_offset)

        urls = url_parse.get_all_urls(base_url, max_crawl_depth=scrape_depth)

        for url, response in urls:
            elem, success = url_parse.process_url(url, response)

            if success:
                fhash = utils.hash_string(url)
                if fhash in hash_to_id_offset:
                    id_offset = hash_to_id_offset[fhash]
                else:
                    id_offset = next_id_offset
                    hash_to_id_offset[fhash] = id_offset
                    next_id_offset += len(elem)
                    new_elements.extend(
                        [
                            (*el, i)
                            for el, i in zip(
                                elem, range(id_offset, id_offset + len(elem))
                            )
                        ]
                    )

                all_elements.extend(
                    [
                        (*el, i)
                        for el, i in zip(elem, range(id_offset, id_offset + len(elem)))
                    ]
                )

            else:
                print("bad html extraction")

        if len(all_elements) == 0:
            raise RuntimeError("Unable to extract data from files.")

        all_samples_df = url_parse.create_train_df(all_elements, id_col=id_col)
        new_samples_df = url_parse.create_train_df(new_elements, id_col=id_col)

        return new_samples_df, all_samples_df, hash_to_id_offset

    def get_context(self, doc_id, context_args) -> str:
        if doc_id not in self.new_samples_df[self.doc_config.id_col]:
            raise ("Document id not in document.")

        if hasattr(context_args, "chunk_radius"):
            chunk_radius = context_args.chunk_radius
            url = self.new_samples_df.loc[
                self.new_samples_df[self.doc_config.id_col] == doc_id
            ]["url"].iat[0]
            source_entries = self.new_samples_df.loc[self.new_samples_df["url"] == url]
            chunks = source_entries["text"].tolist()

            center_chunk_idx = doc_id - self.first_new_id

            window_start = center_chunk_idx - chunk_radius // 2
            window_end = window_start + chunk_radius

            if window_start < 0:
                window_start = 0
                window_end = min(chunk_radius, len(chunks))
            elif window_end > len(chunks):
                window_end = len(chunks)
                window_start = max(0, len(chunks) - chunk_radius)

            window_chunks = chunks[window_start:window_end]
            return "\n".join(window_chunks)

        return ""

    def list_new_sources(self) -> List[str]:
        return self.new_samples_df["url"].unique()

    def get_source_hash_to_id_offset_map(self):
        return self.hash_to_id_offset

    def get_config(self) -> DocumentConfig:
        return self.doc_config

    def get_first_new_id(self) -> int:
        return self.first_new_id

    def get_num_new_ids(self) -> int:
        return self.doc_config.n_new_ids

    def _get_row_with_id(self, id: int) -> int:
        first_id = self.get_first_new_id()
        return self.new_samples_df.iloc[id - first_id]

    def get_reference(self, id: int) -> Reference:
        row = self._get_row_with_id(id)
        return Reference(
            id=row[self.doc_config.id_col], text=row["text"], source=row["url"]
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        # Update file paths to use metadata directory since checkpoints can be
        # used in a different location from where they are first created.
        unsupervised_dataset_name = Path(self.doc_config.unsupervised_dataset).name
        introduction_dataset_name = Path(self.doc_config.introduction_dataset).name
        self.doc_config.unsupervised_dataset = str(
            directory / unsupervised_dataset_name
        )
        self.doc_config.introduction_dataset = str(
            directory / introduction_dataset_name
        )
        self.new_samples_df.to_csv(self.doc_config.introduction_dataset)
        self.all_samples_df.to_csv(self.doc_config.unsupervised_dataset)


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
