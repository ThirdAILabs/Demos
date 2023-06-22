import random
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from thirdai import bolt, bolt_v2

from student import utils
from student.documents import Documents

# InferSamples = List[str]
# Predictions = Sequence[np.ndarray[int]]
# TrainLabels = List[int]
# TrainSamples = List[Tuple[str, TrainLabels]]
InferSamples = List
Predictions = Sequence
TrainLabels = List
TrainSamples = List


class Model:
    def get_model(self) -> bolt.UniversalDeepTransformer:
        raise NotImplementedError()

    def index_documents(
        self,
        documents: Documents,
        on_progress: Callable = lambda **kwargs: None,
        on_freeze_hash_tables: Callable = lambda **kwargs: None,
    ) -> None:
        raise NotImplementedError()

    def forget_documents(self) -> None:
        raise NotImplementedError()

    def searchable(self) -> bool:
        raise NotImplementedError()

    def get_query_col(self) -> str:
        raise NotImplementedError()

    def get_n_ids(self) -> int:
        raise NotImplementedError()

    def get_id_col(self) -> str:
        raise NotImplementedError()

    def get_id_delimiter(self) -> str:
        raise NotImplementedError()

    def train_samples_to_train_batch(self, samples: TrainSamples):
        query_col = self.get_query_col()
        id_col = self.get_id_col()
        id_delimiter = self.get_id_delimiter()
        return [
            {
                query_col: utils.clean_text(text),
                id_col: id_delimiter.join(map(str, labels)),
            }
            for text, labels in samples
        ]

    def balance_train_label_samples(self, samples: TrainSamples, n_samples: int):
        raise NotImplementedError()

    def balance_train_bucket_samples(self, samples: TrainSamples, n_samples: int):
        raise NotImplementedError()

    def infer_samples_to_infer_batch(self, samples: InferSamples):
        query_col = self.get_query_col()
        return [{query_col: utils.clean_text(text)} for text in samples]

    def train_buckets(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        raise NotImplementedError()

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        raise NotImplementedError()

    def train_labels(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        raise NotImplementedError()

    def infer_labels(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        raise NotImplementedError()

    def save_meta(self, directory: Path) -> None:
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()


def unsupervised_train_on_docs(
    model,
    documents: Documents,
    min_epochs: int,
    max_epochs: int,
    metric: str,
    learning_rate: float,
    acc_to_stop: float,
    on_progress: Callable,
    on_freeze_hash_tables: Callable,
):
    config = documents.get_config()
    model._get_model().freeze_hash_tables()
    for i in range(max_epochs):
        metrics = model.cold_start(
            filename=config.unsupervised_dataset,
            strong_column_names=config.strong_cols,
            weak_column_names=config.weak_cols,
            learning_rate=learning_rate,
            epochs=1,
            metrics=[metric],
        )

        val = metrics["train_" + metric][0]
        on_progress(fraction=val)
        if i >= min_epochs - 1 and val > acc_to_stop:
            break


def make_balancing_samples(documents: Documents):
    start_id = documents.get_first_new_id()
    end_id = start_id + documents.get_num_new_ids()
    texts = pd.Series(
        documents.get_reference(id).text() for id in range(start_id, end_id)
    )
    texts = texts.sample(min(25000, len(texts)))
    samples = [
        (query, [i]) for i, passage in texts.items() for query in sent_tokenize(passage)
    ]
    if len(samples) > 25000:
        samples = random.sample(samples, k=25000)
    return samples


class Standard(Model):
    def __init__(
        self,
        id_col="DOC_ID",
        id_delimiter=" ",
        query_col="QUERY",
        input_dim=100_000,
        hidden_dim=512,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_ids = None
        self.model = None
        self.balancing_samples = None

    def get_model(self) -> bolt.UniversalDeepTransformer:
        return self.model

    def get_n_ids(self) -> int:
        return self.n_ids

    def get_query_col(self) -> str:
        return self.query_col

    def get_id_col(self) -> str:
        return self.id_col

    def get_id_delimiter(self) -> str:
        return self.id_delimiter

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def index_documents(
        self,
        documents: Documents,
        on_progress: Callable = lambda **kwargs: None,
        on_freeze_hash_tables: Callable = lambda **kwargs: None,
    ) -> None:
        config = documents.get_config()

        # Standard model always retrains completely.
        self.id_col = config.id_col
        self.n_ids = config.n_new_ids
        self.balancing_samples = make_balancing_samples(documents)

        self.model = bolt.UniversalDeepTransformer(
            data_types={
                self.query_col: bolt.types.text(tokenizer="char-4"),
                self.id_col: bolt.types.categorical(delimiter=self.id_delimiter),
            },
            target=self.id_col,
            n_target_classes=config.n_new_ids,
            integer_target=True,
            options={
                "fhr": self.input_dim,
                "embedding_dimension": self.hidden_dim,
            },
        )

        unsupervised_train_on_docs(
            model=self.model,
            documents=documents,
            min_epochs=10,
            max_epochs=20,
            metric="categorical_accuracy",
            learning_rate=0.005,
            acc_to_stop=0.95,
            on_progress=on_progress,
            on_freeze_hash_tables=on_freeze_hash_tables,
        )

    def forget_documents(self) -> None:
        self.model = None
        self.balancing_samples = None
        self.n_ids = None

    def searchable(self) -> bool:
        return self.model is not None

    def train_labels(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        self.model.train_batch(
            self.train_samples_to_train_batch(samples), learning_rate=learning_rate
        )

    def infer_labels(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        results = self.model.predict_batch(self.infer_samples_to_infer_batch(samples))
        all_predictions = np.argsort(-results, axis=1)  # sorted in descending order
        return all_predictions[:, :n_results]

    def train_buckets(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        self.train_labels(samples=samples, learning_rate=learning_rate, **kwargs)

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        return self.infer_labels(samples=samples, n_results=n_results, **kwargs)

    def balance_train_label_samples(self, samples: TrainSamples, n_samples: int):
        balanced_samples = samples + list(
            random.choices(self.balancing_samples, k=n_samples)
        )
        random.shuffle(balanced_samples)
        return balanced_samples

    def balance_train_bucket_samples(self, samples: List, n_samples: int):
        return self.balance_train_label_samples(samples, n_samples)


class Mach(Model):
    def __init__(
        self,
        id_col="DOC_ID",
        id_delimiter=" ",
        query_col="QUERY",
        input_dim=50_000,
        hidden_dim=2048,
        extreme_output_dim=50_000,
    ):
        self.id_col = id_col
        self.id_delimiter = id_delimiter
        self.query_col = query_col
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.extreme_output_dim = extreme_output_dim
        self.n_ids = 0
        self.model = None
        self.balancing_samples = []

    def get_model(self) -> bolt.UniversalDeepTransformer:
        return self.model

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def get_n_ids(self) -> int:
        return self.n_ids

    def get_query_col(self) -> str:
        return self.query_col

    def get_id_col(self) -> str:
        return self.id_col

    def get_id_delimiter(self) -> str:
        return self.id_delimiter

    def index_documents(
        self,
        documents: Documents,
        on_progress: Callable = lambda **kwargs: None,
        on_freeze_hash_tables: Callable = lambda **kwargs: None,
    ) -> None:
        config = documents.get_config()

        if config.id_col != self.id_col:
            raise ValueError(
                f"Model configured to use id_col={self.id_col}, received document with id_col={config.id_col}"
            )

        if self.model is None:
            self.id_col = config.id_col
            self.model = self.model_from_scratch(documents)
            learning_rate = 0.005
        else:
            if config.n_new_ids > 0:
                doc_id = documents.get_config().id_col
                if doc_id != self.id_col:
                    raise ValueError(
                        f"Document has a different id column ({doc_id}) than the model configuration ({self.id_col})."
                    )
                self.model.introduce_documents(
                    config.introduction_dataset,
                    strong_column_names=config.strong_cols,
                    weak_column_names=[],
                    num_buckets_to_sample=16,
                )
            learning_rate = 0.001

        self.n_ids += config.n_new_ids
        self.add_balancing_samples(documents)

        unsupervised_train_on_docs(
            model=self.model,
            documents=documents,
            min_epochs=10,
            max_epochs=20,
            metric="hash_precision@5",
            learning_rate=learning_rate,
            acc_to_stop=0.95,
            on_progress=on_progress,
            on_freeze_hash_tables=on_freeze_hash_tables,
        )

    def add_balancing_samples(self, documents: Documents):
        samples = make_balancing_samples(documents)
        self.balancing_samples += samples
        if len(self.balancing_samples) > 25000:
            self.balancing_samples = random.sample(self.balancing_samples, k=25000)

    def model_from_scratch(
        self,
        documents: Documents,
    ):
        return bolt.UniversalDeepTransformer(
            data_types={
                self.query_col: bolt.types.text(tokenizer="char-4"),
                self.id_col: bolt.types.categorical(delimiter=self.id_delimiter),
            },
            target=self.id_col,
            n_target_classes=documents.get_config().n_new_ids,
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": self.extreme_output_dim,
                "fhr": self.input_dim,
                "embedding_dimension": self.hidden_dim,
                "rlhf": True,
            },
        )

    def forget_documents(self) -> None:
        if self.model is not None:
            self.model.clear_index()
        self.n_ids = 0
        self.balancing_samples = []

    def searchable(self) -> bool:
        return self.n_ids != 0

    def train_labels(
        self, samples: TrainSamples, learning_rate: float, **kwargs
    ) -> None:
        train_batch = self.train_samples_to_train_batch(samples)
        self.model.train_batch(train_batch, learning_rate=learning_rate)

    def infer_labels(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        self.model.set_decode_params(min(self.n_ids, n_results), min(self.n_ids, 100))
        infer_batch = self.infer_samples_to_infer_batch(samples)
        all_predictions = self.model.predict_batch(infer_batch)
        #####
        return [
            [int(pred) for pred, _ in predictions] for predictions in all_predictions
        ]

    def train_buckets(self, samples: TrainSamples, learning_rate, **kwargs) -> None:
        train_batch = self.train_samples_to_train_batch(samples)
        self.model.train_with_hashes(train_batch, learning_rate=learning_rate)

    def infer_buckets(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        infer_batch = self.infer_samples_to_infer_batch(samples)
        predictions = [
            self.model.predict_hashes(sample)[:n_results] for sample in infer_batch
        ]
        return predictions

    def balance_train_label_samples(self, samples: List, n_samples: int):
        balanced_samples = samples + list(
            random.choices(self.balancing_samples, k=n_samples)
        )
        random.shuffle(balanced_samples)
        return balanced_samples

    def balance_train_bucket_samples(self, samples: List, n_samples: int):
        balancers = utils.random_sample(self.balancing_samples, k=n_samples)
        balancers = [(query, self.get_bucket(labels[0])) for query, labels in balancers]
        balanced_samples = samples + balancers
        random.shuffle(balanced_samples)
        return balanced_samples

    def get_bucket(self, entity: int):
        return self.model.get_index().get_entity_hashes(entity)
