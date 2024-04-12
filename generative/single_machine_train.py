# For further finetuning on downstream task, where it is required to have a context. the format should
# for Bolt-7B
# be {"prompt": <prompt text>, "context": <context text>,"target": <target text>}
# for Bolt-2.5B
# be {"context": <context text>,"target": <target text>}

from thirdai import bolt, dataset, licensing


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
    epochs=1,
    batch_size=256,
    learning_rate=0.0003,
    max_in_memory_batches=20,  # Change as per your memory availablity
    train_metrics=["loss"],
    val_data=val_data,
    val_metrics=["loss"],
)

model.save(SAVE_PATH)
