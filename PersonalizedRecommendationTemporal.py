from thirdai import bolt
import time

TRAIN_FILE = "datasets/movielens_train.csv"
TEST_FILE = "datasets/movielens_test.csv"

model = bolt.UniversalDeepTransformer(
    data_types={
        "userId": bolt.types.categorical(n_unique_classes=6040),
        "movieId": bolt.types.categorical(n_unique_classes=3706),
        "timestamp": bolt.types.date(),
    },
    temporal_tracking_relationships={"userId": ["movieId"]},
    target="movieId",
)

train_config = (bolt.TrainConfig(epochs=3, learning_rate=0.001)
                    .with_metrics(["recall@10"]))

model.train(TRAIN_FILE, train_config)

test_config = (bolt.EvalConfig()
                   .with_metrics(["recall@1", "recall@10", "recall@100"]))

model.evaluate(TEST_FILE, test_config)

start = time.time()

model.predict({"userId": "5825", "timestamp": "2002-04-05"})
model.predict({"userId": "5413", "timestamp": "2003-01-20"})
model.predict({"userId": "1426", "timestamp": "2003-02-04"})
model.predict({"userId": "424", "timestamp": "2003-02-27"})
model.predict({"userId": "4958", "timestamp": "2003-02-28"})

end = time.time()

print("Latency:", (end - start) / 5 * 1000, "ms")