from typing import Callable
import os
import datetime
from pathlib import Path
import pickle
import shutil

from student.models import Model
from student.documents import Documents
from student.loggers import Logger


def pickle_to(obj: object, filepath: Path):
    with open(filepath, "wb") as pkl:
        pickle.dump(obj, pkl)


def unpickle_from(filepath: Path):
    with open(filepath, "rb") as pkl:
        obj = pickle.load(pkl)
    return obj


def default_checkpoint_name():
    return f"checkpoint {datetime.datetime.now()}"


class State:
    def __init__(self) -> None:
        self.model : Model = None
        self.logger : Logger = None
        self.documents : Documents = None
        pass

    def ready(self) -> bool:
        return (
            self.model is not None and 
            self.logger is not None and 
            self.documents is not None and
            self.model.searchable()
        )
    
    def model_pkl_path(directory: Path) -> Path:
        return directory / "model.pkl"
    
    def model_meta_path(directory: Path) -> Path:
        return directory / "model"
    
    def logger_pkl_path(directory: Path) -> Path:
        return directory / "logger.pkl"
    
    def logger_meta_path(directory: Path) -> Path:
        return directory / "logger"
    
    def documents_pkl_path(directory: Path) -> Path:
        return directory / "documents.pkl"
    
    def documents_meta_path(directory: Path) -> Path:
        return directory / "documents"
    
    def save(self, location=default_checkpoint_name(), on_progress: Callable=lambda **kwargs: None):
        total_steps = 8

        # make directory
        directory = Path(location)
        os.mkdir(directory)
        on_progress(current_step=1, total_steps=total_steps)

        # pickle model
        pickle_to(self.model, State.model_pkl_path(directory))
        on_progress(current_step=2, total_steps=total_steps)
        # save model meta
        os.mkdir(State.model_meta_path(directory))
        self.model.save_meta(State.model_meta_path(directory))
        on_progress(current_step=3, total_steps=total_steps)

        # pickle logger
        pickle_to(self.logger, State.logger_pkl_path(directory))
        on_progress(current_step=4, total_steps=total_steps)
        # save logger meta
        os.mkdir(State.logger_meta_path(directory))
        self.logger.save_meta(State.logger_meta_path(directory))
        on_progress(current_step=5, total_steps=total_steps)

        # pickle documents
        pickle_to(self.documents, State.documents_pkl_path(directory))
        on_progress(current_step=6, total_steps=total_steps)
        # save documents meta
        os.mkdir(State.documents_meta_path(directory))
        self.documents.save_meta(State.documents_meta_path(directory))
        on_progress(current_step=7, total_steps=total_steps)

        # zip directory
        zip_file = shutil.make_archive(str(directory), "zip", str(directory))
        shutil.rmtree(directory)
        on_progress(current_step=8, total_steps=total_steps)

        return zip_file

    def load(self, archive: str, on_progress: Callable=lambda **kwargs: None):
        total_steps = 7

        # unzip into directory
        archive_path = Path(archive)
        unpack_dir = archive_path.stem
        shutil.unpack_archive(archive, unpack_dir)
        on_progress(current_step=1, total_steps=total_steps)

        unpack_dir_path = Path(unpack_dir)

        # load model
        self.model = unpickle_from(State.model_pkl_path(unpack_dir_path))
        on_progress(current_step=2, total_steps=total_steps)
        self.model.load_meta(State.model_meta_path(unpack_dir_path))
        on_progress(current_step=3, total_steps=total_steps)

        # load logger
        self.logger = unpickle_from(State.logger_pkl_path(unpack_dir_path))
        on_progress(current_step=4, total_steps=total_steps)
        self.logger.load_meta(State.logger_meta_path(unpack_dir_path))
        on_progress(current_step=5, total_steps=total_steps)

        # load documents
        self.documents = unpickle_from(State.documents_pkl_path(unpack_dir_path))
        on_progress(current_step=6, total_steps=total_steps)
        self.documents.load_meta(State.documents_meta_path(unpack_dir_path))
        on_progress(current_step=7, total_steps=total_steps)



