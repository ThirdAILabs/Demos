# BOLT-7B

BOLT-7B is meticulously trained on CPUs, employing dynamic sparse technology, which lies at the core of our groundbreaking BOLT engine. A decade of dedicated research has culminated in BOLT, ensuring unparalleled efficiency for neural networks. The dynamic sparsity feature empowers us to selectively activate neural pathways, enabling optimal training even on CPU resources.

This release have 7.5 billion parameter model, along with both inference and training scripts tailored for distributed as well as single machine training scenarios.

[Medium Blog]()

## Model
BOLT-7B is a 7.5 billion parameter model, trained on a corpus of 120 billion tokens extracted from the C4 dataset. The training task focused on next word prediction, with a context length of 2048 tokens.
We trained model in distributed setting using 30 intel sapphire rapids (Intel Sapphire Rapids dual socket with 56 cores per socket).

## Quick Start
BOLT provides simple apis via our python package. Please register for a free user license at [https://www.thirdai.com/try-bolt/](https://www.thirdai.com/try-bolt/).

You can also try out the model on hugging face spaces [here](https://huggingface.co/spaces/thirdai/LLM-7B-open-qa). 

Steps to start using `BOLT-7B`:
1. Install thirdai.
```commandline
pip install thirdai
```
2. Download the model.
```commandline
wget https://thirdai-corp-public.s3.us-east-2.amazonaws.com/Bolt-7B/bolt-7b
```
3. Try out one of the scripts here, or use it for your own application!

#### Note: 
To run `BOLT-7B` you need a machine with 30GB of RAM to load the model for inference only. Finetuning requires roughly (depending on dataset size) 130GB of RAM. 

### Model Generation

This function generates predictions using the specified parameters.

#### Parameters

- `input_tokens (List[int])`: List of input tokens.
- `prompt (List[int])`: List of prompt tokens.`
- `max_predictions (int)`: Maximum number of predictions to generate.
- `beam_width (int)`: Width of the beam search.
- `temperature (Optional[float])`: Temperature parameter for controlling randomness in predictions. Defaults to `None` meaning temperature sampling is not used.

#### Example Usage

```python
model.generate(
    input_tokens=input_tokens,
    prompt_tokens=prompt_tokens,
    max_predictions=100,
    beam_width=3,
    temperature=1.2,
)
```

### Model Training

This method is used to train a machine learning model using the specified training data and hyperparameters.

#### Parameters

- `train_data (required)`: The training data to be used for training the model.
- `learning_rate (optional, default: 1e-5)`: The learning rate used by the optimizer for updating the model's parameters.
- `epochs (optional, default: 5)`: The number of epochs (iterations over the entire training dataset) to train the model.
- `batch_size (optional, default: 10,000)`: The number of samples per batch during training.
- `train_metrics (optional, default: ["loss"])`: A list of metrics to be computed and displayed during training.
- `val_data (optional, default: None)`: The validation data to be used for evaluating the model's performance after each epoch.
- `val_metrics (optional, default: [])`: A list of metrics to be computed and displayed during validation.
- `max_in_memory_batches (optional, default: None)`: The number of batches to load in memory at once during training. 

#### Example Usage

```python
model.train(
    train_data=train_data_path,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    train_metrics=["loss"],
    val_data=val_data_path,
    val_metrics=["loss"],
    max_in_memory_batches=10,
)
```


## Data Processing

The `LLMDataSource` expects data in a format where each line is a json object with the keys `prompt`, `target` and `context`. The `prompt` field specifies the initial prompt to model for generation. The `target` field specifies the text that should be generated. The `context` specifies additional tokens that will be prepended to the target tokens before training, however the `context` tokens will not be generated or used as labels. For example:

```json
{"prompt": "open_qa", "context": "How are you?", "target": "I am fine."}
{"prompt": "next word prediction", "context": "Order a pizza", "target": "for the party tonight."}
```
