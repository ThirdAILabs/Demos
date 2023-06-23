from utils import CSV as NCSV

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

    from student.loggers import InMemoryLogger
    from student.state import State

    state = State(model, logger=InMemoryLogger())
    state.documents = doclist

    state.save(checkpoint_location)

def neural_db_to_playground(db, checkpoint_location, csv = None):
    db_model = db._savable_state.model
    query_col = db_model.get_query_col()
    id_col = db_model.get_id_col()
    id_delimiter = db_model.get_id_delimiter()
    raw_model = db_model.get_model()
    model_input_dim = db_model.input_dim
    model_hidden_dim = db_model.hidden_dim
    model_extreme_output_dim = db_model.extreme_output_dim

    if csv is None:
        raw_model.clear_index()
    raw_model._get_model().freeze_hash_tables()

    from student import models

    model = models.Mach(
        id_col=id_col,
        id_delimiter=id_delimiter,
        query_col=query_col,
        input_dim=model_input_dim,
        hidden_dim=model_hidden_dim,
        extreme_output_dim=model_extreme_output_dim,
    )

    model.model = raw_model

    from student.documents import DocList, CSV

    doclist = DocList()

    if csv is not None:
        if not isinstance(csv, NCSV):
            raise ValueError("Invalid type for CSV. Must be a CSV object.")
        if id_col != csv.id_column:
            raise ValueError(f"To export to playground, the CSV ID column must match the underlying bolt model ('{id_col}')")
        doc = CSV(
            path=csv.path,
            id_col=csv.id_column,
            strong_cols=csv.strong_columns,
            weak_cols=csv.weak_columns,
            display_cols=csv.reference_columns,
        )

        doclist = DocList()
        doclist.add_document(doc)

        model.n_ids = doc.get_num_new_ids()

        model.balancing_samples = models.make_balancing_samples(doc)
    

    from student.loggers import InMemoryLogger
    from student.state import State

    state = State(model, logger=InMemoryLogger())
    state.documents = doclist


    state.save(checkpoint_location)

