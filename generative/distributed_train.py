import ray
from ray.air import RunConfig, FailureConfig, CheckpointConfig, session
from ray.train.torch import TorchConfig

import thirdai
from thirdai import bolt, data, dataset, licensing
import thirdai.distributed_bolt as dist

# Please reach out to us at https://www.thirdai.com/contact/ to receive a free
# license for finetuning.
licensing.activate("")

TRAIN_FILE = ""  # add the trianing file here
VALIDATION_FILE = ""  # add the validatoin file here
MODEL_PATH = ""  # add model_path here


# This scaling config is right now for single machine training, and to run num_workers worker on that machine
def setup_ray(num_workers=2):
    import ray
    import thirdai.distributed_bolt as dist
    from ray.air import ScalingConfig

    # reserve one CPU for Ray Trainer
    num_cpu_per_node = (dist.get_num_cpus() - 1) // num_workers

    assert num_cpu_per_node >= 1, "Number of CPUs per node should be greater than 0"

    ray.init(
        runtime_env={
            "env_vars": {"OMP_NUM_THREADS": f"{num_cpu_per_node}"},
        },
        ignore_reinit_error=True,
    )
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=False,
        resources_per_worker={"CPU": num_cpu_per_node},
        placement_strategy="PACK",
    )
    return scaling_config


def train_loop_per_worker(config):
    # only train data would be splitted
    train_split_data_iterator = session.get_dataset_shard("train")
    validation_data_iterator = session.get_dataset_shard("validation")

    thirdai.logging.setup(log_to_stderr=False, path="log.txt", level="info")

    model = None
    if session.get_world_rank() == 0:
        model = bolt.GenerativeModel.load(MODEL_PATH)

    # broadcast the model
    model = dist.prepare_model(model)

    train_data = dataset.RayTextDataSource(
        train_split_data_iterator, should_tokenize=True
    )
    val_data = dataset.RayTextDataSource(validation_data_iterator, should_tokenize=True)

    model.train_distributed(
        train_data=train_data,
        epochs=config["epochs"],
        batch_size=config["batch_size_per_worker"],
        learning_rate=config["learning_rate"],
        train_metrics=["loss"],
        val_data=val_data,
        val_metrics=["loss"],
    )

    # Make sure to pass absolute path here, else it would just save the model inside <Home>/ray_results/Bolt_Trainer_<session-id>/BoltTrainer-<session-id>/rank_0/trained_generative.model
    if session.get_world_rank() == 0:
        model.save("/share/pratik/trained_generative.model")


scaling_config = setup_ray()

# parallelism doesnt matter since we are reading from just one file
train_ray_ds = ray.data.read_text(TRAIN_FILE, parallelism=1)
validation_ray_ds = ray.data.read_text(VALIDATION_FILE, parallelism=1)

trainer = dist.BoltTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "learning_rate": 0.001,
        "epochs": 3,
        "batch_size_per_worker": 20_000,
    },
    scaling_config=scaling_config,
    backend_config=TorchConfig(backend="gloo"),
    run_config=RunConfig(
        failure_config=FailureConfig(max_failures=-1),
        checkpoint_config=CheckpointConfig(num_to_keep=1),
    ),
    datasets={"train": train_ray_ds, "validation": validation_ray_ds},
)

result_checkpoint_and_history = trainer.fit()

ray.shutdown()
