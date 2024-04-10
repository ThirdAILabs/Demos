# PUBMED

## PUBMED DESCRIPTION

PubMed comprises more than 36 million citations for biomedical literature from MEDLINE, life science journals, and online books. Citations may include links to full-text content from PubMed Central and publisher web sites.

We are using Title and Abstract from the pubmed dataset. You can find the dataset to download from here https://huggingface.co/datasets/pubmed and also can refer in https://www.ncbi.nlm.nih.gov/search/all/?term=pubmed

## PROCESS OF INFLATION OF PUBMED DATA

For each data point the abstract is large, so we chunked a large abstract into 3 smaller chuncks with few sentences by using nltk sentence tokenizer and made that 35MM data points to roughly 120MM data points.

## DATASET FORMAT

The Sample data of the the Pubmed120M is given in the folder

1. Unsupervised File : pubmed_1000_unsup.csv

   the columns of the data "id,para_id,label,title,abstract"

2. Supervised File: pubmed_1000_sup.csv

   the columns of the data "QUERY,id"

## TRAINING SCRIPT

Training File: train_pubmed.py

For you to run the training file, You need to some set of things prior.

1. Setup Nomad cluster and get the base_url for the the cluster to submit the train job.

   Follow https://github.com/ThirdAILabs/neuraldb-enterprise for setting up Nomad cluster on various machines.

2. Install latest version of thirdai and its dependcies.

   pip3 install thirdai[neural_db]

3. Get your license key from thirdai, For more information look into https://www.thirdai.com/contact/

## Training Infra used by ThirdAI

Our neural db enterprise is installed on 6 AMD servers with AMD EPYC 9754 128-Core Processor. It took 60 hours to train this model.


## Infrence 

Our model have efficient inference of ~400ms/query.

We are hosting this on an AMD Bergamo machine 255 core and 1 TB mem with 4 instances of model to reach 20 QPS. Note we are not compute bound.

To reach 20 QPS : 32 core and 250 gig memory * 4 instaces are required.
NeuralDB Enterprise hosting perform autoscalling as per load.

Try now: http://70.233.60.118:3000/