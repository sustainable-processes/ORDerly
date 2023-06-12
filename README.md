# ORDerly: Standardised cleaning of chemical reactions

This repository is the official implementation of [ORDerly](https://figshare.com/articles/dataset/ORDerly_chemical_reactions_condition_benchmarks/23298467). 

<img src="images/abstract_fig.png" alt="Abstract Figure" width="300">

Use ORDerly to:
- Access the [ORDerly benchmark dataset](https://figshare.com/articles/dataset/ORDerly_chemical_reactions_condition_benchmarks/23298467) for reaction condition prediction.
- Extract, clean, and train a model on USPTO data in ORD format using default arguments.
- Apply customised extraction and cleaning operations to your own data proprietary data in ORD format.
- Reproduce results from the paper. 

<!--
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
-->





# Simple setup
Using default arguments.

## ORDerly
### Installation

`pip install orderly`

### Download data from the publically available ORD database 

`orderly download`
This will create a folder called `/data/ord/` in your current directory, and download the data into `ord/`

Alternatively, you can also follow the instructions on the [official website](https://github.com/open-reaction-database/ord-data) to download the data in ```ord-data/data/```.

### Extract data from your ORD files

`orderly extract`

If you want to run ORDerly on your own data, and want to specify the input and output path:

`orderly extract --input_path="/data/ord/" --output_path="/data/orderly/"`

This will generate a parquet file for each ORD file.

### Cleaning the data

`orderly clean`

This will produce train and test parquet files, along with a .json file showing the arguments used and a .log file showing the operations run.

## Training a condition prediction algorithm with this data

For this, clone the repository and use the makefile.

### Requirements
Python dependencies can be installed via ```poetry``` from within the `orderly/condition_prediction` folder:

- run in terminal: ```poetry install```

### Train model






# Customisable setup






## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```






## Download ORD data
The data should be placed within `data/ord`

### Using the ORD database

We want to download the ORD data locally, this can be done through any of the following methods:

- Follow the instructions at: https://github.com/open-reaction-database/ord-data, we specifically care about the folders in ```ord-data/data/```
- Docker install with linux (run in terminal):
    ```
    make linux_download_ord
    ``` 
- Docker install with mac (run in terminal):
    ```
    make root_download_ord
    make sudo_chown
    ```


### Using your own data in ORD format


## Train

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
























### 2. Run extraction

We can run extraction using: ```poetry run python -m orderly.extract```. Using ```poetry run python -m orderly.extract --help``` will explain the arguments. Certain args must be set such as data paths.

### 3. Run cleaning

We can run cleaning using: ```poetry run python -m orderly.clean```. Using ```poetry run python -m orderly.clean --help``` will explain the arguments. Certain args must be set such as data paths.

