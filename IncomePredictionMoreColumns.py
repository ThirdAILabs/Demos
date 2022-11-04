from thirdai import bolt
import time

TRAIN_FILE = "datasets/census_income_train.csv"
TEST_FILE = "datasets/census_income_test.csv"

model = bolt.UniversalDeepTransformer(
    data_types={
        "age": bolt.types.numerical(range=(17, 90)),
        "workclass": bolt.types.categorical(n_unique_classes=9),
        "fnlwgt": bolt.types.numerical(range=(12285, 1484705)),
        "education": bolt.types.categorical(n_unique_classes=16),
        "education-num": bolt.types.categorical(n_unique_classes=16),
        "marital-status": bolt.types.categorical(n_unique_classes=7),
        "occupation": bolt.types.categorical(n_unique_classes=15),
        "relationship": bolt.types.categorical(n_unique_classes=6),
        "race": bolt.types.categorical(n_unique_classes=5),
        "sex": bolt.types.categorical(n_unique_classes=2),
        "capital-gain": bolt.types.numerical(range=(0, 99999)),
        "capital-loss": bolt.types.numerical(range=(0, 4356)),
        "hours-per-week": bolt.types.numerical(range=(1, 99)),
        "native-country": bolt.types.categorical(n_unique_classes=42),
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

start = time.time()

model.predict({'age': '36', 'workclass': 'Federal-gov', 'fnlwgt': '212465', 'education': 'Bachelors', 'education-num': '13', 'marital-status': 'Married-civ-spouse', 'occupation': 'Adm-clerical', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': '0', 'capital-loss': '0', 'hours-per-week': '40', 'native-country': 'United-States', 'label': '<=50K'})
model.predict({'age': '26', 'workclass': 'Private', 'fnlwgt': '82091', 'education': 'HS-grad', 'education-num': '9', 'marital-status': 'Never-married', 'occupation': 'Adm-clerical', 'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Female', 'capital-gain': '0', 'capital-loss': '0', 'hours-per-week': '39', 'native-country': 'United-States', 'label': '<=50K'})
model.predict({'age': '58', 'workclass': '?', 'fnlwgt': '299831', 'education': 'HS-grad', 'education-num': '9', 'marital-status': 'Married-civ-spouse', 'occupation': '?', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': '0', 'capital-loss': '0', 'hours-per-week': '35', 'native-country': 'United-States', 'label': '<=50K'})
model.predict({'age': '48', 'workclass': 'Private', 'fnlwgt': '279724', 'education': 'HS-grad', 'education-num': '9', 'marital-status': 'Married-civ-spouse', 'occupation': 'Machine-op-inspct', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': '3103', 'capital-loss': '0', 'hours-per-week': '48', 'native-country': 'United-States', 'label': '>50K'})
model.predict({'age': '43', 'workclass': 'Private', 'fnlwgt': '346189', 'education': 'Masters', 'education-num': '14', 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': '0', 'capital-loss': '0', 'hours-per-week': '50', 'native-country': 'United-States', 'label': '>50K'})

end = time.time()

print("Latency:", (end - start) / 5 * 1000, "ms")