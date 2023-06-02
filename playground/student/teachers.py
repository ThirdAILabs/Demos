from typing import Sequence
import math
import random

import pandas as pd
from nltk.tokenize import sent_tokenize

from student.models import Model
from student.loggers import Logger
from student import utils


# TODO(Geordie): BalancedModel wrapper

def association_training_samples(
    model: Model, 
    text_a: str, text_b: str, top_k: int, n_samples: int
):
    # Based on Yash's suggestion to chunk target phrase if it is long.
    b_buckets = model.infer_buckets(sent_tokenize(text_b), n_results=top_k)
    samples = [(text_a, buckets) for buckets in b_buckets]
    return utils.random_sample(samples, k=n_samples)

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def associate(
    model: Model, 
    logger: Logger,
    text_a: str, 
    text_b: str, 
    top_k: int, 
    jaccard_threshold: float,
):
    train_samples = association_training_samples(model, text_a, text_b, top_k, n_samples=16)
    jaccard = 0
    for _ in range(3):
        balanced_train_samples = model.balance_train_bucket_samples(train_samples, n_samples=50)
        model.train_buckets(balanced_train_samples, learning_rate=0.001)
        new_a_buckets, new_b_buckets = model.infer_buckets([text_a, text_b], n_results=top_k)
        jaccard = jaccard_similarity(set(new_a_buckets), set(new_b_buckets))
    
    logger.log(session_id="global", action="associate", args={
        "text_a": text_a,
        "text_b": text_b,
        "top_k": top_k,
        "jaccard_threshold": jaccard_threshold
    })

    return jaccard

def associate_new(
    model: Model, 
    logger: Logger,
    text_a: str, 
    text_b: str, 
    top_k: int, 
    jaccard_threshold: float,
):
    model.model.associate({"query":text_a}, {"query":text_b},top_k)
    
    logger.log(session_id="global", action="associate", args={
        "text_a": text_a,
        "text_b": text_b,
        "top_k": top_k,
        "jaccard_threshold": jaccard_threshold
    })

    return 0

def associate_ui(get_model, get_logger, strength_k_map=[1, 3, 5]):
    import gradio as gr

    def run_associate(text_a: str, text_b: str, strength: float):
        strength = math.ceil(strength)
        top_k = strength_k_map[strength-1]

        associate(get_model(), get_logger(), text_a, text_b, top_k, jaccard_threshold=0.9)
        # associate_new(get_model(), get_logger(), text_a, text_b, top_k, jaccard_threshold=0.9)
        return gr.Markdown.update(f"Success.", visible=True)

    
    gr.Markdown("## Associate phrases")
    with gr.Row():
        with gr.Column():
            text_a = gr.Text(label="From")
        with gr.Column():
            text_b = gr.Text(label="To")

    strength_slider = gr.Slider(1, 3, value=3, label="Strength")
    associate_btn = gr.Button("Associate")
    associate_msg = gr.Markdown("", visible=False)
    associate_btn.click(
        run_associate, 
        inputs=[text_a, text_b, strength_slider], 
        outputs=[associate_msg], 
        show_progress=True
    )

    return [text_a, text_b, associate_msg]


def contextual_upweight_training_samples(
    model: Model,
    context: str, to_upweight: str, weight: int, n_results: int, n_samples: int
):
    if to_upweight not in context:
        return "Word to upweight cannot be found in context."
    upweight_pos = context.find(to_upweight)
    augmented_context_pieces = (
        [context[:upweight_pos]] + 
        ([to_upweight] * (weight - 1)) + 
        [context[upweight_pos + len(to_upweight):]]
    )
    augmented_context = ' '.join(augmented_context_pieces)
    labels = model.infer_buckets([augmented_context], n_results=n_results)[0]
    return [
        (context, random.sample(labels, len(labels)))
        for _ in range(n_samples)
    ]


def contextual_upweight(
    model: Model, logger: Logger,
    context: str, to_upweight: str, weight: int, n_results: int,
):
    n_samples = 3
    train_samples = contextual_upweight_training_samples(model, context, to_upweight, weight, n_results, n_samples)
    lr = 0.01
    for _ in range(4):
        balanced_samples = model.balance_train_bucket_samples(train_samples, n_samples * 2)
        model.train_buckets(balanced_samples, learning_rate=lr)
        lr = max(0.01, lr * 0.5)

    logger.log(session_id="global", action="contextual_upweight", args={
        "context": context,
        "to_upweight": to_upweight,
        "weight": weight,
        "n_results": n_results,
    })


def contextual_upweight_ui(get_model, get_logger, n_results: int):
    import gradio as gr

    def run_contextual_upweight(to_upweight: str, context: str):
        weight = 5
        error = contextual_upweight(get_model(), get_logger(), context, to_upweight, weight, n_results)
        return gr.Markdown.update(error if error else "Success.", visible=True)
    
    gr.Markdown("## Emphasize")
    with gr.Row():
        with gr.Column():
            context = gr.Text(label="Text")
        with gr.Column():
            to_upweight = gr.Text(label="Keywords to emphasize in text")
    emphasize_btn = gr.Button("Emphasize")
    emphasize_error = gr.Markdown("", visible=False) 
    emphasize_btn.click(
        fn=run_contextual_upweight, 
        inputs=[to_upweight, context], 
        outputs=[emphasize_error], 
        show_progress=True
    )


def upweight_training_samples(
    model: Model,
    to_upweight: str, references: pd.Series, n_samples: int, n_words_per_sample: int, n_results: int,
):
    related_passages = [passage for passage in references if to_upweight in passage]
    related_words = [word for passage in related_passages for word in passage.split(' ')]
    if len(related_words) == 0:
        return "Word to upweight cannot be found in reference files."
    labels = model.infer_buckets([to_upweight], n_results=n_results)[0]
    train_samples = [
        ' '.join([to_upweight] + [random.choice(related_words) for _ in range(n_words_per_sample)])
        for _ in range(n_samples)
    ]
    return [(sample, labels) for sample in train_samples]


def upweight(
    model: Model, logger: Logger,
    to_upweight: str, references: pd.Series, n_samples: int, n_words_per_sample: int, n_results: int
):
    train_samples = upweight_training_samples(model, to_upweight, references, n_samples, n_words_per_sample, n_results)
    balanced_samples = model.balance_train_bucket_samples(train_samples, n_samples)
    model.train_buckets(balanced_samples, learning_rate=0.001)

    logger.log(session_id="global", action="upweight", args={
        "to_upweight": to_upweight,
        "n_samples": n_samples,
        "n_words_per_sample": n_words_per_sample,
    })
    

def upweight_ui(get_model, get_logger, get_references, n_results):
    import gradio as gr

    def run_upweight(to_upweight: str):
        n_upweight_samples = 5
        n_words_per_upweight_sample = 10
        error = upweight(get_model(), get_logger(), to_upweight, get_references(), n_upweight_samples, n_words_per_upweight_sample, n_results)
        return gr.Markdown.update(error if error else "Success.", visible=True)

    gr.Markdown("## Focus on keywords")
    to_upweight = gr.Text(label="Focus Keywords")
    upweight_btn = gr.Button("Focus")
    upweight_error = gr.Markdown("", visible=False) 
    upweight_btn.click(
        run_upweight, 
        inputs=[to_upweight], 
        outputs=[upweight_error], 
        show_progress=True
    )


def upvote(model: Model, logger: Logger, query: str, liked_passage_id: int, n_results=int):
    train_samples = [(query, [liked_passage_id])] * 16
    for _ in range(3):
        balanced_samples = model.balance_train_label_samples(train_samples, n_samples=50)
        model.train_labels(balanced_samples, learning_rate=0.001)
    logger.log(
        session_id="global",
        action="upvote",
        args={"query": query, "liked_passage_id": liked_passage_id},
    )

def like_button(
    get_model, get_logger, 
    query: str, search_result_ids: Sequence[int], id_pos: int, n_results: int
):
    import gradio as gr

    def like(query):
        upvote(get_model(), get_logger(), query, search_result_ids[id_pos], n_results=n_results)
        return gr.Button.update("Retraining successful! Search again to see the difference 👍")
    
    like_btn = gr.Button("👍")
    like_btn.click(like, inputs=[query], outputs=[like_btn])
    return like_btn

def like_button_update():
    import gradio as gr
    return gr.Button.update("👍")
