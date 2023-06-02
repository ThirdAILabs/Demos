from typing import Optional
from pathlib import Path
import pandas as pd


class Logger:
    def log(self, session_id: str, action: str, args: dict, train_samples: Optional[any]=None):
        raise NotImplementedError()
    
    def save_meta(self, directory: Path):
        raise NotImplementedError()
    
    def load_meta(self, directory: Path):
        raise NotImplementedError()


class InMemoryLogger(Logger):
    def make_log(session_id=[], action=[], args=[], train_samples=[]):
        return pd.DataFrame({
            "session_id": session_id,
            "action": action,
            "args": args,
            "train_samples": train_samples,
        })
    
    def __init__(self, logs = make_log()):
        self.logs = logs

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        self.logs = pd.concat([
            self.logs,
            InMemoryLogger.make_log(
                session_id=[session_id],
                action=[action],
                args=[args],
                train_samples=[train_samples],
            )
        ])

    def save_meta(self, directory: Path):
        pass
    
    def load_meta(self, directory: Path):
        pass


class NoOpLogger(Logger):
    def __init__(self) -> None:
        pass

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        pass

    def save_meta(self, directory: Path):
        pass
    
    def load_meta(self, directory: Path):
        pass