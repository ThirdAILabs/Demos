import os
import pandas as pd
import zipfile
import json
import numpy as np


def _download_dataset(url, zip_file, check_existence, output_dir):
    if not os.path.exists(zip_file):
        os.system(f"curl {url} --output {zip_file}")

    if any([not os.path.exists(must_exist) for must_exist in check_existence]):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)


def to_batch(dataframe):
    return [
        {col_name: str(col_value) for col_name, col_value in record.items()}
        for record in dataframe.to_dict(orient="records")
    ]


def download_movielens():
    MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    MOVIELENS_ZIP = "./movielens.zip"
    MOVIELENS_DIR = "./movielens"
    RATINGS_FILE = MOVIELENS_DIR + "/ml-1m/ratings.dat"
    MOVIE_TITLES = MOVIELENS_DIR + "/ml-1m/movies.dat"
    TRAIN_FILE = "./movielens_train.csv"
    TEST_FILE = "./movielens_test.csv"
    SPLIT = 0.9
    INFERENCE_BATCH_SIZE = 5

    _download_dataset(
        url=MOVIELENS_1M_URL,
        zip_file=MOVIELENS_ZIP,
        check_existence=[RATINGS_FILE, MOVIE_TITLES],
        output_dir=MOVIELENS_DIR,
    )

    df = pd.read_csv(RATINGS_FILE, header=None, delimiter="::", engine="python")
    df.columns = ["userId", "movieId", "rating", "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    movies_df = pd.read_csv(
        MOVIE_TITLES,
        header=None,
        delimiter="::",
        engine="python",
        encoding="ISO-8859-1",
    )
    movies_df.columns = ["movieId", "movieTitle", "genre"]
    movies_df["movieTitle"] = movies_df["movieTitle"].apply(
        lambda x: x.replace(",", "")
    )

    df = pd.merge(df, movies_df, on="movieId")
    df = df[["userId", "movieTitle", "timestamp"]]
    df = df.sort_values("timestamp")

    n_train_samples = int(SPLIT * len(df))
    train_df = df.iloc[:n_train_samples]
    test_df = df.iloc[n_train_samples:]
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    index_batch = to_batch(df.iloc[:INFERENCE_BATCH_SIZE])
    inference_batch = to_batch(df.iloc[:INFERENCE_BATCH_SIZE][["userId", "timestamp"]])

    return TRAIN_FILE, TEST_FILE, inference_batch, index_batch


def download_clinc():
    CLINC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip"
    CLINC_ZIP = "./clinc150_uci.zip"
    CLINC_DIR = "./clinc"
    MAIN_FILE = CLINC_DIR + "/clinc150_uci/data_full.json"
    TRAIN_FILE = "./clinc_train.csv"
    TEST_FILE = "./clinc_test.csv"
    INFERENCE_BATCH_SIZE = 5

    _download_dataset(
        url=CLINC_URL,
        zip_file=CLINC_ZIP,
        check_existence=[MAIN_FILE],
        output_dir=CLINC_DIR,
    )

    samples = json.load(open(MAIN_FILE))

    train_samples = samples["train"]
    test_samples = samples["test"]

    train_text, train_category = zip(*train_samples)
    test_text, test_category = zip(*test_samples)

    train_df = pd.DataFrame({"text": train_text, "category": train_category})
    test_df = pd.DataFrame({"text": test_text, "category": test_category})

    train_df["text"] = train_df["text"].apply(lambda x: x.replace(",", ""))
    test_df["text"] = test_df["text"].apply(lambda x: x.replace(",", ""))

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    inference_batch = to_batch(
        test_df[["text"]].sample(frac=1).iloc[:INFERENCE_BATCH_SIZE]
    )

    return TRAIN_FILE, TEST_FILE, inference_batch


def download_criteo():
    CRITEO_URL = "http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz"
    CRITEO_ZIP = "./criteo.tar.gz"
    CRITEO_DIR = "./criteo"
    MAIN_FILE = CRITEO_DIR + "/train.txt"
    CREATED_TRAIN_FILE = "./criteo/train_udt.csv"
    CREATED_TEST_FILE = "./criteo/test_udt.csv"

    os.makedirs("./criteo", exist_ok=True)
    
    if not os.path.exists(CRITEO_ZIP):
        print(
            f"Downloading from {CRITEO_URL}. This can take 20-40 minutes depending on the Criteo server."
        )
        os.system(f"wget -t inf -c {CRITEO_URL} -O {CRITEO_ZIP}")

    if not os.path.exists(MAIN_FILE):
        print("Extracting files. This can take up to 10 minutes.")
        os.system(
            f"tar -xvzf {CRITEO_ZIP} -C {CRITEO_DIR}"
        )

    df = pd.read_csv(MAIN_FILE, delimiter="\t", header=None)
    n_train = int(0.8 * df.shape[0])
    header = (
        ["label"]
        + [f"num_{i}" for i in range(1, 14)]
        + [f"cat_{i}" for i in range(1, 27)]
    )

    print("Processing the dataset (this will take about 6-7 mins).")
    min_vals = df.iloc[:, 1:14].min()
    df.iloc[:, 1:14] = np.round(np.log(df.iloc[:, 1:14] - min_vals + 1), 2)
    min_vals = np.float32(df.iloc[:, 1:14].min())
    max_vals = np.float32(df.iloc[:, 1:14].max())
    y = np.float32(df.iloc[:, 0])
    n_unique_classes = list(df.iloc[:, 14:].nunique())

    y_train = y[:n_train]
    y_test = y[n_train:]

    if not os.path.exists(CREATED_TRAIN_FILE) or not os.path.exists(CREATED_TEST_FILE):
        print("saving the train and test datasets (this will take about 10 mins)")
        df[:n_train].to_csv(CREATED_TRAIN_FILE, header=header, index=False)
        df[n_train:].to_csv(CREATED_TEST_FILE, header=header, index=False)

    df_sample = df.iloc[n_train : n_train + 2]
    df_sample = df_sample.fillna("")
    sample_batch = [
        {header[i]: str(df_sample.iloc[0, i]) for i in range(1, 40)}
    ]  # first sample
    sample_batch.append(
        {header[i]: str(df_sample.iloc[1, i]) for i in range(1, 40)}
    )  # second sample

    return (
        CREATED_TRAIN_FILE,
        CREATED_TEST_FILE,
        y_train,
        y_test,
        min_vals,
        max_vals,
        n_unique_classes,
        sample_batch,
    )


def prep_fraud_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df["amount"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]).abs()

    def upsample(df):
        fraud_samples = df[df["isFraud"] == 1]
        upsampling_ratio = 5
        for i in range(upsampling_ratio):
            df = pd.concat([df, fraud_samples], axis=0)
        return df

    df = upsample(df)

    df = df.sample(frac=1)

    SPLIT = 0.8
    n_train_samples = int(SPLIT * len(df))
    train_df = df.iloc[:n_train_samples]
    test_df = df.iloc[n_train_samples:]

    train_filename = "fraud_dataset/new_train.csv"
    test_filename = "fraud_dataset/new_test.csv"

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    INFERENCE_BATCH_SIZE = 5
    inference_batch = to_batch(
        df.iloc[:INFERENCE_BATCH_SIZE][
            [
                "step",
                "type",
                "amount",
                "nameOrig",
                "oldbalanceOrg",
                "newbalanceOrig",
                "nameDest",
                "oldbalanceDest",
                "newbalanceDest",
                "isFlaggedFraud",
            ]
        ]
    )

    return train_filename, test_filename, inference_batch


def download_census_income():
    CENSUS_INCOME_BASE_DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    TRAIN_FILE = "./census_income_train.csv"
    TEST_FILE = "./census_income_test.csv"
    COLUMN_NAMES = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    INFERENCE_BATCH_SIZE = 5
    if not os.path.exists(TRAIN_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.data --output {TRAIN_FILE}"
        )
        # reformat the train file
        with open(TRAIN_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TRAIN_FILE, "w") as file:
            # Write header
            file.write(','.join(COLUMN_NAMES) + '\n')
            # Convert ", " delimiters to ",".
            file.writelines([line.replace(", ", ",") for line in data[1:]])

    if not os.path.exists(TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )
        # reformat the test file
        with open(TEST_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TEST_FILE, "w") as file:
            # Write header
            file.write(','.join(COLUMN_NAMES) + '\n')
            # Convert ", " delimiters to ",".
            # Additionally, for some reason each of the labels end with a "." in the test set
            # loop through data[1:] since the first line is bogus
            file.writelines([line.replace(".", "").replace(", ", ",") for line in data[1:]])
    
    n_lines = 0
    lines_for_inference_batch = []
    for line in open(TEST_FILE, "r"):
        if n_lines == 0:
            n_lines += 1
            continue

        if n_lines == INFERENCE_BATCH_SIZE + 1:
            break

        lines_for_inference_batch.append(line)
        n_lines += 1
    
    inference_batch = [
        {
            col_name: value 
            for col_name, value in zip(COLUMN_NAMES, line.split(','))
        } 
        for line in lines_for_inference_batch
    ]

    return TRAIN_FILE, TEST_FILE, inference_batch
