# FEDn example - Sentiment analysis on Swedish reviews with BERT  
This repository contains an example FEDn client to train a Swedish BERT model for sentiment analysis on Swedish reviews.
The model is implemented using Pytorch and Huggingface Transformers. 

> This example has only been tested in a pseudo-distributed environment. Performed tests are shown at the bottom of this page.

> Note that this example shows how to configure FEDn for training, and how to configure and start clients. It is assumed that a FEDn network is aleady up and running with a blank, unconfigured Reducer. If this is not the case, start here: https://github.com/scaleoutsystems/fedn/blob/master/README.md

### Download and prepare data
This example use the Swedish reviews dataset from https://huggingface.co/datasets/swedish_reviews. 
To download and prepare the partitioned dataset, run:
``` bash
python3 create_data_partitions.py [N_PARTITIONS]
```
The default number of partitions is 2.
### Create the compute package
Create the compute package by running:
```bash
sh ./create_compute_package.sh
```
To clear the system and set a new compute package, see: https://github.com/scaleoutsystems/fedn/blob/master/docs/FAQ.md

For an explaination of the compute package structure and content: https://github.com/scaleoutsystems/fedn/blob/develop/docs/tutorial.md
 
### Create the initial model
The baseline BERT model is specified in the file 'client/models/bert_pytorch_model.py'.
Install the dependencies in 'client/requirements.txt' and run
```bash
sh ./init_model.sh
```
to create an initial model. This script creates a Swedish BERT model for sentiment analysis (positive or negative) and serializes the model to a .npz-file.
The base BERT model to use, [KB/bert-base-swedish-case](https://huggingface.co/KB/bert-base-swedish-cased) or [KB/albert-base-swedish-cased-alpha](https://huggingface.co/KB/albert-base-swedish-cased-alpha), can be configured in the 'client/settings.yaml' file.

## Configuring the Reducer

Navigate to 'https://localhost:8090' (or the url of your Reducer) and follow instructions to upload the compute package in 'package/compute_package.tar.gz' and the initial model in 'initial_model/initial_model.npz'. 

## Attaching a client to the federation

1. First, download 'client.yaml' from the Reducer 'Network' page, and replace the content in your local 'client.yaml'. 
2. Start a client. Here there are different options (see below): 
    - Docker 
    - docker-compose
    - [Native client (OSX/Linux)](https://github.com/scaleoutsystems/examples/tree/main/how-tos/start-native-fedn-client)

#### Docker
1. Build the image

``` bash
docker build . -t bert-client:latest
```

2. Start a client (edit the path of the volume mounts to provide the absolute path to your local folder.)
```
docker run -v /absolute-path-to-this-folder/data/:/app/data:ro -v /absolute-path-to-this-folder/client.yaml:/app/client.yaml --network fedn_default bert-client fedn run client -in client.yaml 
```
(Repeat above steps as needed to deploy additional clients).

#### docker-compose
To start 2 clients, run: 

```bash
docker-compose -f docker-compose.yaml -f private-network.yaml up
```
The number of clients can be configured in docker-compose.yaml.
> If you are connecting to a Reducer part of a distributed setup or in Studio, you should omit 'private-network.yaml'. 

#### Native client on OSX/Linux
The compute package assumes that the local dataset is in a folder 'data' in the same folder as you start the client. Make a new folder and copy the data partition you want to use into data:
```bash
cp data/clients/0/*.csv data/
```

### Start training 
When clients are running, navigate to the 'Control' page of the Reducer to start the training. 

### Configuring the client
Some settings are possible to configure to vary the conditions for the training. These configurations are expsosed in the file 'client/settings.yaml': 

```yaml
# Base BERT model
base_model: "KB/albert-base-swedish-cased-alpha"
# Maximum number of tokens in each review
max_text_length: 35
# Fraction of client data to use for training
nr_training_samples: 50
# Fraction of client data to use for testing
nr_test_samples: 50
# Batch size
batch_size: 8
# Nr of epochs
epochs: 1
```
These default settings can be used as a small test. For better performance, increase number of samples and text length limit and/or use the "KB/bert-base-swedish-cased" model.
### Test results
Below are two test results using BERT and ALBERT from @Kungbib. Note that these models are quite big (BERT ~500MB and ALBERT ~50MB) and can require some time to train, especially without GPU.
Take this into consideration when selecting the size of your experiment, i.e. number of clients, amount of data, length of text samples, number of training rounds, etc. Support for CUDA is implemented but this has not been tested.

Below two tests have been executed using 2 clients and 3 training rounds.
#### Test 1
Settings:
```yaml  
base_model: "KB/albert-base-swedish-cased-alpha"
max_text_length: 50
nr_training_samples: 300
nr_test_samples: 100
batch_size: 16
epochs: 1
```
Results:
![Albert results](https://i.ibb.co/pxmrdC3/albertres.png)

#### Test 2
Settings:
```yaml  
base_model: "KB/bert-base-swedish-cased"
max_text_length: 35
nr_training_samples: 250
nr_test_samples: 100
batch_size: 8
epochs: 1
```
Results:
![Bert results](https://i.ibb.co/mrjfpvF/bertres.png)