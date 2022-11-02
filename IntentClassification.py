from thirdai import bolt
import time

TRAIN_FILE = "datasets/clinc_train.csv"
TEST_FILE = "datasets/clinc_test.csv"

model = bolt.UniversalDeepTransformer(
    data_types={
        "text": bolt.types.text(),
        "category": bolt.types.categorical(n_unique_classes=150)
    },
    target="category"
)

train_config = (bolt.TrainConfig(epochs=5, learning_rate=0.01)
                    .with_metrics(["categorical_accuracy"]))

model.train(TRAIN_FILE, train_config)

test_config = (bolt.EvalConfig()
                   .with_metrics(["categorical_accuracy"]))

model.evaluate(TEST_FILE, test_config)

start = time.time()
for _ in range(100):
    model.predict({"text": "what expression would i use to say i love you if i were an italian"})
end = time.time()

print("Latency:", (end - start) / 100 * 1000, "ms")