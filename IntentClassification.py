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

model.predict({"text": "tell me what the gas mileage is on my car"})
model.predict({"text": "what expression would i use to say i love you if i were an italian"})
model.predict({"text": "help me pick a new location to travel to"})
model.predict({"text": "please put dusting on my list of things to do"})
model.predict({"text": "what's the tire pressure of my tires"})

end = time.time()

print("Latency:", (end - start) / 5 * 1000, "ms")