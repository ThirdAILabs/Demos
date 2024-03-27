from thirdai.neural_db import ModelBazaar, neural_db as ndb
import thirdai

# Find the hash of the converted unsupervised file to pass it in the reference for supervised file.
csv_doc = ndb.CSV(
    "./pubmed_1000_unsup.csv",
    id_column="id",
    strong_columns = ["title"],
    weak_columns = ["abstract"],
    reference_columns = ["title", "abstract"]
)

thirdai.licensing.activate("YOUR-LICENSE-KEY")

bazaar = ModelBazaar(base_url="YOUR-ENTERPRISE-URL") # Enter your neuraldb enterprise URL here, e.g. "http://11.22.33.44/api/". Be sure to include /api/ at the end of the URL.

bazaar.log_in(email="YOUR-EMAIL", password="YOUR-PASSWORD")

train_extra_options = {
    "num_models_per_shard": 2,  # How many models you want to train on per each data shard.
    "num_shards": 2,  # How many shards do you want the data to be sharded.
    "allocation_memory": 500, # How much MB of RAM you want to allocate for the data sharding.
    "model_cores": 2, # How many cores you want to use on each model training
    "model_memory": 6800, # how much memory you want to allocate for each model training.
    "csv_id_column": "id",
    "csv_strong_columns": ["title"],
    "csv_weak_columns": ["abstract"],
    "csv_reference_columns": ["title", "abstract"],
    "csv_query_column": "QUERY",
    "epochs": 10
}

model_name = "pubmed1k"

model = bazaar.train(
    model_name=model_name,
    unsupervised_docs=["./pubmed_1000_unsup.csv"],
    supervised_docs=[("./pubmed_1000_sup.csv", str(csv_doc.hash))],
    doc_type="local", # disk of data files, either on disk local to this script or in the Nomad cluster nfs directory
    sharded=True,
    is_async=True,
    train_extra_options=train_extra_options,
)
