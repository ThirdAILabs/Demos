from thirdai import bolt
import time

TRAIN_FILE = "datasets/census_income_train.csv"
TEST_FILE = "datasets/census_income_test.csv"

model = bolt.UniversalDeepTransformer(
    data_types={
        "age": bolt.types.numerical(range=(17, 90)),
        "workclass": bolt.types.categorical(n_unique_classes=9),
        "education": bolt.types.categorical(n_unique_classes=16),
        "label": bolt.types.categorical(n_unique_classes=2),
    },
    target="label"
)

train_config = (bolt.TrainConfig(epochs=5, learning_rate=0.01)
                    .with_metrics(["categorical_accuracy"]))

model.train(TRAIN_FILE, train_config)

test_config = (bolt.EvalConfig()
                   .with_metrics(["categorical_accuracy"]))

model.evaluate(TEST_FILE, test_config)