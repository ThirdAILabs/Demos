# PII Detection Streaming Service

ThirdAI's service to predict named entity tags on large corpus of data in a streaming service.

You can run the following service either on your local machine, or you can run on remote machines.

## Requirements

1. Ensure you have docker daemon running and be able to pull docker images on all machines you want to run. docker version has to be > 25.0.3, please update to the latest version if you have older versions.

2. Currently we support local files or files from S3.

3. You need to have a `.env` file. Please checkout the .env.example to see how to configure the service.

## Running Locally

To run the service locally, execute the following command

`` ./run.sh path_to_env_file ``

This will do the following
   - Use the specified `.env` file for configuration.
   - Set up logging, and the logs will be stored in  `./logs`.
   - Pull the docker image and run the container locally.


## Running on Remote Machines

To run the service on remote machines, you need to provide additional arguments: username, machine IPs.

`` ./run.sh path_to_env_file username machine_ips ``

### Example

`` ./run.sh .env yash 192.168.1.1,192.168.1.2 ``

1. Ensure that you can able to ssh into these remote machines from the current machine.
2. You can pull docker images in these machines (Make sure you are added to the docker group).
3. If you are using a local folder path, make sure that all the remote machines can access that folder (Put the folder in a NFS directory) if you are using more than one remote machines for scale.
4. Ensure you have ``numactl`` installed on linux machines. To install you can run ``sudo apt install numactl``.

## Logs

if ypu run locally your logs will be in ``./logs`` and if you run using remote machines you will have your logs at ``/tmp/docker_logs`` on each remote machine.

In the folder you will find so many files of this format
1. `tika_log_test.txt`, this file will contain files parsed and how many sentences are extracted from each file and the amount of time taken for that.
2. `{file_name}_{time}.csv`, all the csv files will have a file_name stating the log correspond to that file, in each csv file we will have sentence and the corresponding ner tags.