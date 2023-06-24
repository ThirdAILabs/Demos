from thirdai import neural_db as ndb
from pathlib import Path
import pandas as pd
import shutil

class CSV(ndb.Document):
    def __init__(self, path, id_column, strong_columns, weak_columns, reference_columns) -> None:
        self.df = pd.read_csv(path)
        self.df = self.df.sort_values(id_column)
        assert len(self.df[id_column].unique()) == len(self.df[id_column])
        assert self.df[id_column].min() == 0
        assert self.df[id_column].max() == len(self.df[id_column]) - 1

        for col in strong_columns + weak_columns + reference_columns:
            self.df[col] = self.df[col].fillna("")

        self.path = Path(path)
        self._hash = ndb.utils.hash_file(path)
        self.id_column = id_column
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns
    
    def hash(self) -> str:
        return self._hash
    
    def size(self) -> int:
        return len(self.df)
    
    def name(self) -> str:
        return self.path.name
    
    def strong_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.strong_columns])
    
    def weak_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([row[col] for col in self.weak_columns])
    
    def reference(self, element_id: int) -> ndb.Reference:
        row = self.df.iloc[element_id]
        text = " ".join([row[col] for col in self.reference_columns])
        return ndb.Reference(
            document=self,
            element_id=element_id, 
            text=text, 
            source=str(self.path.absolute()), 
            metadata=row.to_dict())
    
    def context(self, element_id: int, radius) -> str:
        rows = self.df.iloc[
            max(0, element_id - radius):
            min(len(self.df), element_id + radius)]
        return " ".join([row[col] for col in self.reference_columns for row in rows])
    
    def save_meta(self, directory: Path):
        # Let's copy the original CSV file to the provided directory
        shutil.copy(self.path, directory)
    
    def load_meta(self, directory: Path):
        # Since we've moved the CSV file to the provided directory, let's make
        # sure that we point to this CSV file.
        self.path = directory / self.path.name

