from thirdai import neural_db as ndb
from pathlib import Path
import pandas as pd
import shutil
from thirdai import bolt, licensing

class CSV(ndb.Document):
    def __init__(self, path, id_column, strong_columns, weak_columns, reference_columns) -> None:
        self.df = pd.read_csv(path)
        self.df = self.df.sort_values(id_column)
        assert len(self.df[id_column].unique()) == len(self.df[id_column])
        assert self.df[id_column].min() == 0
        assert self.df[id_column].max() == len(self.df[id_column]) - 1

        for col in strong_columns + weak_columns:
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
            metadata={})
    
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


def bolt_and_csv_to_checkpoint(
    raw_model,
    csv_location,
    checkpoint_location,
    model_input_dim = 50000,
    model_hidden_dim = 2048,
    id_col = "DOC_ID",
    query_col = "QUERY",
    id_delimiter = ":",
    strong_cols = ["passage"],
    weak_cols = ["para"],
    display_cols = ["passage"],
):
    # raw_model = bolt.UniversalDeepTransformer.load(bolt_location)
    raw_model._get_model().freeze_hash_tables()
    
    from student import models

    model = models.Mach(
        id_col=id_col,
        id_delimiter=id_delimiter,
        query_col=query_col,
        input_dim=model_input_dim,
        hidden_dim=model_hidden_dim,
    )

    model.model = raw_model

    from student.documents import CSV, DocList

    doc = CSV(
        path=csv_location,
        id_col=id_col,
        strong_cols=strong_cols,
        weak_cols=weak_cols,
        display_cols=display_cols,
    )

    doclist = DocList()
    doclist.add_document(doc)

    model.n_ids = doc.get_num_new_ids()

    model.balancing_samples = models.make_balancing_samples(doc)

    from student.loggers import InMemoryLogger, LoggerList
    from student.state import State

    state = State(model, logger=InMemoryLogger())
    state.documents = doclist

    state.save(checkpoint_location)

def bolt_to_checkpoint(
    raw_model,
    checkpoint_location,
    model_input_dim = 50000,
    model_hidden_dim = 2048,
    id_col = "DOC_ID",
    query_col = "QUERY",
    id_delimiter = ":",
):
    # raw_model = bolt.UniversalDeepTransformer.load(bolt_location)
    raw_model.clear_index()
    raw_model._get_model().freeze_hash_tables()

    from student import models

    model = models.Mach(
        id_col=id_col,
        id_delimiter=id_delimiter,
        query_col=query_col,
        input_dim=model_input_dim,
        hidden_dim=model_hidden_dim,
    )

    model.model = raw_model

    from student.documents import CSV, DocList

    doclist = DocList()

    from student.loggers import InMemoryLogger, LoggerList
    from student.state import State

    state = State(model, logger=InMemoryLogger())
    state.documents = doclist

    state.save(checkpoint_location)

