import datetime
import os
import pickle
import shutil
from pathlib import Path
from typing import Callable

from student.documents import DocList
from student.loggers import Logger
from student.models import Model


def pickle_to(obj: object, filepath: Path):
    with open(filepath, "wb") as pkl:
        pickle.dump(obj, pkl)


def unpickle_from(filepath: Path):
    with open(filepath, "rb") as pkl:
        obj = pickle.load(pkl)
    return obj


def default_checkpoint_name():
    return Path(f"checkpoint {datetime.datetime.now()}.3ai")


class State:
    def __init__(self, model: Model, logger: Logger) -> None:
        self.model = model
        self.logger = logger
        self.documents = DocList()
        pass

    def ready(self) -> bool:
        return (
            self.model is not None
            and self.logger is not None
            and self.documents is not None
            and self.model.searchable()
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

    def save(
        self,
        location=default_checkpoint_name(),
        on_progress: Callable = lambda **kwargs: None,
    ):
        total_steps = 7

        # make directory
        directory = Path(location)
        os.mkdir(directory)
        on_progress(fraction=1 / total_steps)

        # pickle model
        pickle_to(self.model, State.model_pkl_path(directory))
        on_progress(fraction=2 / total_steps)
        # save model meta
        os.mkdir(State.model_meta_path(directory))
        self.model.save_meta(State.model_meta_path(directory))
        on_progress(fraction=3 / total_steps)

        # pickle logger
        pickle_to(self.logger, State.logger_pkl_path(directory))
        on_progress(fraction=4 / total_steps)
        # save logger meta
        os.mkdir(State.logger_meta_path(directory))
        self.logger.save_meta(State.logger_meta_path(directory))
        on_progress(fraction=5 / total_steps)

        # pickle documents
        pickle_to(self.documents, State.documents_pkl_path(directory))
        on_progress(fraction=6 / total_steps)
        # save documents meta
        os.mkdir(State.documents_meta_path(directory))
        self.documents.save_meta(State.documents_meta_path(directory))
        on_progress(fraction=7 / total_steps)

        return str(directory)

    def load(location: Path, on_progress: Callable = lambda **kwargs: None):
        state = State(model=None, logger=None)

        total_steps = 6

        # load model
        state.model = unpickle_from(State.model_pkl_path(location))
        on_progress(fraction=1 / total_steps)
        state.model.load_meta(State.model_meta_path(location))
        on_progress(fraction=2 / total_steps)

        # load logger
        state.logger = unpickle_from(State.logger_pkl_path(location))
        on_progress(fraction=3 / total_steps)
        state.logger.load_meta(State.logger_meta_path(location))
        on_progress(fraction=4 / total_steps)

        # load documents
        state.documents = unpickle_from(State.documents_pkl_path(location))
        on_progress(fraction=5 / total_steps)
        state.documents.load_meta(State.documents_meta_path(location))
        on_progress(fraction=6 / total_steps)

        return state
