from typing import List, Callable, Union
from pathlib import Path
import shutil
import time
import os
import copy

import pandas as pd

from student import utils
from parsing_utils import pdf_parse, doc_parse

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
    
class Documents:
    def list_sources(self) -> List[str]:
        raise NotImplementedError()

    def get_source_hash_to_id_offset_map(self):
        raise NotImplementedError()

    def get_config(self) -> DocumentConfig:
        raise NotImplementedError()
    
    def get_n_new_ids(self) -> int:
        raise NotImplementedError()

    # index of dataframe is expected to be the same as ids
    def get_new_samples_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError()
    
    # index of series is expected to be the same as ids
    def get_new_references(self) -> pd.Series:
        raise NotImplementedError()
    
    # index of series is expected to be the same as ids
    def get_new_display_items(self) -> pd.Series:
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()


# Base class for PDF and DOCX classes because they share the same logic.
class Extracted(Documents):
    def __init__(self, 
        files: List[str],
        expected_id_col: str,
        hash_to_id_offset = {},
        next_id_offset: int = 0,
        on_extract_success: Callable = lambda **kwargs: None,
        on_extract_failed: Callable = lambda **kwargs: None,
        on_extraction_complete: Callable = lambda **kwargs: None,
    ):
        if len(files) == 0:
            raise ValueError("Received empty files list.")
        
        self.source_files = files
        train_file = f"train.csv"
        intro_file = f"intro.csv"

        self.new_samples_df, self.all_samples_df, self.hash_to_id_offset = self.process_data(
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
        files: List[str], id_col: str, hash_to_id_offset: dict, next_id_offset: int,
        on_extract_success: Callable, on_extract_failed: Callable,
        on_extraction_complete: Callable,
    ) -> pd.DataFrame:
        raise NotImplementedError()
    
    def list_sources(self) -> List[str]:
        return self.source_files
    
    def get_source_hash_to_id_offset_map(self):
        return self.hash_to_id_offset
        
    def get_config(self) -> DocumentConfig:
        return self.doc_config
    
    def get_n_new_ids(self) -> int:
        return self.doc_config.n_new_ids
    
    def get_new_samples_dataframe(self) -> pd.DataFrame:
        return self.new_samples_df
    
    def get_new_references(self) -> pd.Series:
        return self.new_samples_df["passage"]
    
    def get_new_display_items(self) -> pd.Series:
        return self.new_samples_df["display"]
    
    def display_entry(self, index) -> str:
        return super().display_entry(index)
    
    def save_meta(self, directory: Path):
        # Save source files
        for file in self.source_files:
            shutil.copy(file, directory)

    def load_meta(self, directory: Path):
        # Update file paths to use metadata directory since checkpoints can be
        # used in a different location from where they are first created.
        self.source_files = [str(directory / Path(file).name) for file in self.source_files]
        unsupervised_dataset_name = Path(self.doc_config.unsupervised_dataset).name
        introduction_dataset_name = Path(self.doc_config.introduction_dataset).name
        self.doc_config.unsupervised_dataset = str(directory / unsupervised_dataset_name)
        self.doc_config.introduction_dataset = str(directory / introduction_dataset_name)
        self.new_samples_df.to_csv(self.doc_config.introduction_dataset)
        self.all_samples_df.to_csv(self.doc_config.unsupervised_dataset)
    
        

class PDF(Extracted):
    def __init__(
        self, 
        files: List[str], 
        expected_id_col: str,
        hash_to_id_offset = {},
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
        files: List[str], id_col: str, hash_to_id_offset: dict, next_id_offset: int,
        on_extract_success: Callable, on_extract_failed: Callable,
        on_extraction_complete: Callable,
    ) -> pd.DataFrame:
        # Extract elements from each file
        bad_files = []
        all_elements = []
        new_elements = []
        hash_to_id_offset = copy.deepcopy(hash_to_id_offset)
        
        for f in files :
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
                    new_elements.extend([(*el, i) for el, i in zip(elem, range(id_offset, id_offset + len(elem)))])
                all_elements.extend([(*el, i) for el, i in zip(elem, range(id_offset, id_offset + len(elem)))])
            

            else :
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
        hash_to_id_offset = {},
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
        files: List[str], id_col: str, hash_to_id_offset: dict, next_id_offset: int,
        on_extract_success: Callable, on_extract_failed: Callable,
        on_extraction_complete: Callable,
    ):
        # Extract elements from each file
        bad_files = []
        all_elements = []
        new_elements = []
        hash_to_id_offset = copy.deepcopy(hash_to_id_offset)
        
        for f in files :
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
                    new_elements.extend([(*el, i) for el, i in zip(elem, range(id_offset, id_offset + len(elem)))])
                all_elements.extend([(*el, i) for el, i in zip(elem, range(id_offset, id_offset + len(elem)))])
                    
            else :
                on_extract_failed(file=f)
                bad_files.append(f)

        if len(all_elements) == 0:
            raise RuntimeError("Unable to extract data from files.")
        
        on_extraction_complete(elements=all_elements, bad_files=bad_files)
        all_samples_df = doc_parse.create_train_df(all_elements, id_col=id_col)
        new_samples_df = doc_parse.create_train_df(new_elements, id_col=id_col)
    
        return new_samples_df, all_samples_df, hash_to_id_offset


class CSV(Documents):
    def __init__(self, path, id_col, strong_cols, weak_cols, display_cols, id_offset=0):
        if id_offset > 0:
            raise ValueError("CSV does not support id_offset.")
        
        self.path = path
        self.df = pd.read_csv(self.path)
        for col in strong_cols + weak_cols + display_cols:
            self.df[col] = self.df[col].fillna("")

        self.display = self.df.apply(
            lambda row: '\n'.join(row[display_cols]), axis=1
        )
        self.references = self.df.apply(
            lambda row: '\n'.join(row[strong_cols + weak_cols]), axis=1
        )
        self.doc_config = DocumentConfig(
            id_col=id_col,
            n_new_ids=len(self.df),
            unsupervised_dataset=path,
            introduction_dataset=path,
            strong_cols=strong_cols,
            weak_cols=weak_cols,
            hash=utils.hash_file(self.path)
        )

        
    def list_sources(self) -> List[str]:
        return [self.path]
    
    def get_source_hash_to_id_offset_map(self):
        return {self.doc_config.hash: 0}

    def get_config(self) -> DocumentConfig:
        return self.doc_config
    
    def get_n_new_ids(self) -> int:
        return self.doc_config.n_new_ids
    
    def get_new_samples_dataframe(self) -> pd.Series:
        return self.df
    
    def get_new_references(self) -> pd.Series:
        return self.references
    
    def get_new_display_items(self) -> pd.Series:
        return self.display
    
    def save_meta(self, directory: Path):
        pass
    
    def load_meta(self, directory: Path):
        pass


class DocList(Documents):
    def __init__(self, id_offset=0):
        if id_offset > 0:
            raise ValueError("CSV does not support id_offset.")
        
        self.docs = []
        self.df = None
        self.references = None
        self.display = None
        self.id_col = None
    
    def list_sources(self) -> List[str]:
        return [src for doc in self.docs for src in doc.list_sources()]
    
    def get_source_hash_to_id_offset_map(self):
        return {
            fhash: id_offset 
            for doc in self.docs
            for fhash, id_offset in doc.get_source_hash_to_id_offset_map().items()
        }
    
    def add_document(self, document: Documents):
        new_doc_id_col = document.get_config().id_col
        if self.id_col is not None and new_doc_id_col != self.id_col:
            raise ValueError("Tried to add document with mismatched id column to doc list.")

        new_doc_df = document.get_new_samples_dataframe() 
        if (new_doc_df[new_doc_id_col] < self.get_n_new_ids()).any():
            raise ValueError("Tried to add document with invalid id offset to doc list.")
        
        expected_begin = self.get_n_new_ids()
        expected_end = expected_begin + len(new_doc_df)
        if not all([new_doc_df[new_doc_id_col].iloc[i] == i + self.get_n_new_ids() for i in range(len(new_doc_df))]):
            raise ValueError(f"Expected document ids to be consecutive integers in [{expected_begin},{expected_end})")
        
        self.df = new_doc_df if self.df is None else pd.concat([self.df, new_doc_df])
        self.references = (
            document.get_new_references() 
            if self.references is None 
            else pd.concat([self.references, document.get_new_references()])
        )
        self.display = (
            document.get_new_display_items() 
            if self.display is None
            else pd.concat([self.display, document.get_new_display_items()])
        )

        self.docs.append(document)

    def get_n_new_ids(self):
        return sum([doc.get_n_new_ids() for doc in self.docs])
    
    def clear(self):
        self.docs = []
        self.df = None
        self.references = None
        self.display = None
        self.id_col = None
    
    def get_new_samples_dataframe(self) -> pd.Series:
        return self.df
    
    def get_new_references(self) -> pd.Series:
        return self.references
    
    def get_new_display_items(self) -> pd.Series:
        return self.display
    
    def save_meta(self, directory: Path):
        for i, doc in enumerate(self.docs):
            subdir = directory / str(i)
            os.mkdir(subdir)
            doc.save_meta(subdir)
    
    def load_meta(self, directory: Path):
        for i, doc in enumerate(self.docs):
            subdir = directory / str(i)
            doc.load_meta(subdir)
