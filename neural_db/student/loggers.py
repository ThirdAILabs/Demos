import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


class Logger:
    def name(self):
        raise NotImplementedError()

    def log(
        self,
        session_id: str,
        action: str,
        args: dict,
        train_samples: Optional[any] = None,
    ):
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()


class InMemoryLogger(Logger):
    def make_log(session_id=[], action=[], args=[], train_samples=[]):
        return pd.DataFrame(
            {
                "session_id": session_id,
                "action": action,
                "args": args,
                "train_samples": train_samples,
            }
        )

    def __init__(self, logs=make_log()):
        self.logs = logs

    def name(self):
        return "in_memory"

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        self.logs = pd.concat(
            [
                self.logs,
                InMemoryLogger.make_log(
                    session_id=[session_id],
                    action=[action],
                    args=[args],
                    train_samples=[train_samples],
                ),
            ]
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


class DBLogger(Logger):
    from DB import models as db_models
    from DB.db import get_session

    def __init__(self):
        # TODO(Yash) Initialize anything else as necessary.
        # E.g. would it be better to move the engine initialization here so
        # we can store all relevant metadata like whether it's local or online,
        # or the url.
        pass

    def name(self):
        return "db"

    def log(
        self,
        session_id: str,
        action: str,
        args: dict,
        train_samples: Optional[any] = None,
    ):
        rlhf_data = DBLogger.db_models.Action(
            user_id=session_id,
            action=action,
            args=args,
            train_samples={"train_samples": train_samples},
        )

        with DBLogger.get_session() as session:
            session.add(rlhf_data)
            session.commit()
            session.refresh(rlhf_data)

    def save_meta(self, directory: Path):
        # TODO(Yash): if the database is local (e.g. sqlite?) move the
        # corresponding files into this directory.
        pass

    def load_meta(self, directory: Path):
        # TODO(Yash): perform any necessary setup
        pass


class LoggerList(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def name(self):
        return "list"

    def log(
        self,
        session_id: str,
        action: str,
        args: dict,
        train_samples: Optional[any] = None,
    ):
        [
            logger.log(
                session_id=session_id,
                action=action,
                args=args,
                train_samples=train_samples,
            )
            for logger in self.loggers
        ]

    def save_meta(self, directory: Path):
        for logger in self.loggers:
            os.mkdir(directory / logger.name())
            logger.save_meta(directory / logger.name())

    def load_meta(self, directory: Path):
        for logger in self.loggers:
            logger.load_meta(directory / logger.name())


class NoOpLogger(Logger):
    def __init__(self) -> None:
        pass

    def name(self):
        return "no_op"

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        pass

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass
