# This script is for further pretraining, where each datapoint is {"target": <text>}, see sample dataset
# For further finetuning on downstream task, where it is required to have a context. the format should
# be {"context": <context text>,"target": <target text>}

from thirdai import bolt, dataset, licensing


# Please register for a free license at https://www.thirdai.com/try-bolt/
licensing.activate("")

train_filename = ""  # train file path
val_filename = ""  # validation file path
MODEL_PATH = ""
SAVE_PATH = ""

train_data = dataset.LLMDataSource(train_filename)
val_data = dataset.LLMDataSource(val_filename)

model = bolt.GenerativeModel.load(MODEL_PATH)

model.train(
    train_data=train_data,
    epochs=5,
    batch_size=10_000,
    learning_rate=0.00001,
    train_metrics=["loss"],
    val_data=val_data,
    val_metrics=["loss"],
)

model.save(SAVE_PATH)
