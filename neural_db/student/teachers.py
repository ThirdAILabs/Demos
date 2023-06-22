import math
import random
from typing import Sequence

import pandas as pd
from nltk.tokenize import sent_tokenize

from DB import models as db_models
from DB.db import get_session
from student import utils
from student.loggers import Logger
from student.models import Model


def association_training_samples(
    model: Model, text_a: str, text_b: str, top_k: int, n_samples: int
):
    # Based on Yash's suggestion to chunk target phrase if it is long.
    b_buckets = model.infer_buckets(sent_tokenize(text_b), n_results=top_k)
    samples = [(text_a, buckets) for buckets in b_buckets]
    return utils.random_sample(samples, k=n_samples)


def associate(
    model: Model,
    logger: Logger,
    user_id: str,
    text_a: str,
    text_b: str,
    top_k: int,
):
    train_samples = association_training_samples(
        model, text_a, text_b, top_k, n_samples=16
    )
    for _ in range(3):
        balanced_train_samples = model.balance_train_bucket_samples(
            train_samples, n_samples=50
        )
        model.train_buckets(balanced_train_samples, learning_rate=0.001)

    logger.log(
        session_id=user_id,
        action="associate",
        args={
            "text_a": text_a,
            "text_b": text_b,
            "top_k": top_k,
        },
    )

    rlhf_data = db_models.Action(
        user_id=user_id,
        action="associate",
        args={
            "text_a": text_a,
            "text_b": text_b,
            "top_k": top_k,
        },
    )

    with get_session() as session:
        session.add(rlhf_data)
        session.commit()


def upvote(
    model: Model, logger: Logger, user_id: str, query: str, liked_passage_id: int
):
    train_samples = [(query, [liked_passage_id])] * 16
    for _ in range(3):
        balanced_samples = model.balance_train_label_samples(
            train_samples, n_samples=50
        )
        model.train_labels(balanced_samples, learning_rate=0.001)
    logger.log(
        session_id=user_id,
        action="upvote",
        args={"query": query, "liked_passage_id": liked_passage_id},
    )

    rlhf_data = db_models.Action(
        user_id=user_id,
        action="upvote",
        args={"query": query, "liked_passage_id": str(liked_passage_id)},
    )
    with get_session() as session:
        session.add(rlhf_data)
        session.commit()
