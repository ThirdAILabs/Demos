# BOLT2.5B

BOLT2.5B is meticulously trained on CPUs, employing dynamic sparse technology, which lies at the core of our groundbreaking BOLT engine. A decade of dedicated research has culminated in BOLT, ensuring unparalleled efficiency for neural networks. The dynamic sparsity feature empowers us to selectively activate neural pathways, enabling optimal training even on CPU resources.

This release have 2.5 billion parameter model, along with both inference and training scripts tailored for distributed as well as single machine training scenarios.

## Model
BOLT2.5 is a 2.5 billion parameter model, trained on a corpus of 40 billion tokens extracted from the C4 dataset. The training task focused on next word prediction, with a context length of 512 tokens.
We trained model in distributed setting using 10 intel sapphire rapids (Intel Sapphire Rapids dual socket with 56 cores per socket).

## Quick Start
BOLT provides simple apis via our python package. A free user license that is limited to model inference is provided along with the example scripts. Please reach out to us at https://www.thirdai.com/contact/ to receive a free license for finetuning. 

You can also try out the model on hugging face spaces here. 

Steps to start using `BOLT2.5B`:
1. Install thirdai, at least version 0.7.20.
```commandline
pip install thirdai
```
2. Download the model.
```commandline
wget https://storage.googleapis.com/bolt-llm/bolt2.5-generative.model
```
3. Try out one of the scripts here, or use it for your own application!

#### Note: 
To run `BOLT2.5B` you need a machine with 10Gb of RAM to load the model for inference only. Finetuning requires roughly (depending on dataset size) 40Gb of RAM. 

### Model Generation

This function generates predictions using the specified parameters.

#### Parameters

- `input_tokens (List[int])`: List of input tokens.
- `max_predictions (int)`: Maximum number of predictions to generate.
- `beam_width (int)`: Width of the beam search.
- `temperature (Optional[float])`: Temperature parameter for controlling randomness in predictions. Defaults to `None` meaning temperature sampling is not used.

#### Example Usage

```python
model.generate(
    input_tokens=input_tokens,
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

#### Example Usage

```python
model.train(
    train_data=train_data_path,
    epochs=10,
    batch_size=8_000,
    learning_rate=0.0001,
    train_metrics=["loss"],
    val_data=val_data_path,
    val_metrics=["loss"]
)
```


## Data Processing

The `LLMDataSource` expects data in a format where each line is a json object with the key `target`, and optionally the key `context`. The `target` field specifies the text that should be generated. The `context` specifies additional tokens that will be prepended to the target tokens before featurization, however the `context` tokens will not be generated or used as labels. For example:

```json
{"context": "How are you?", "target": "I am fine."}
{"context": "Order a pizza", "target": "for the party tonight."}
```
