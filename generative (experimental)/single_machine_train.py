# For further finetuning on downstream task, where it is required to have a context. the format should
# for Bolt-7B
# be {"prompt": <prompt text>, "context": <context text>,"target": <target text>}
from thirdai import bolt, dataset, licensing
import argparse

parser = argparse.ArgumentParser(description="Model Finetuning Parameters")


parser.add_argument(
    "--learning_rate",
    type=float,
    required=True,
    help="Learning rate for the model training",
)
parser.add_argument(
    "--epochs", type=int, required=True, help="Number of epochs for the model training"
)
parser.add_argument(
    "--batch_size", type=int, required=True, help="Batch size per worker"
)

args = parser.parse_args()

# Please register for a free license at https://www.thirdai.com/try-bolt/
licensing.activate("")

train_filename = ""
val_filename = ""
MODEL_PATH = ""
SAVE_PATH = ""

train_data = dataset.LLMDataSource(train_filename)
val_data = dataset.LLMDataSource(val_filename)

model = bolt.GenerativeModel.load(MODEL_PATH)

model.train(
    train_data=train_data,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    max_in_memory_batches=10,  # Change as per your memory availablity
    train_metrics=["loss"],
    val_data=val_data,
    val_metrics=["loss"],
)

model.save(SAVE_PATH)
