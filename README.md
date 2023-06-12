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


# Download the ORDerly benchmark dataset
```
pip install orderly

orderly.download_benchmark()
```


# Simple setup (default arguments)
Using default arguments.

## ORDerly
### Installation

```pip install orderly```

### Download data from the publically available ORD database 

```orderly download```
This will create a folder called ```/data/ord/``` in your current directory, and download the data into ```ord/```

Alternatively, you can also follow the instructions on the [official website](https://github.com/open-reaction-database/ord-data) to download the data in ```ord-data/data/```.

### Extract data from your ORD files

```orderly extract```

If you want to run ORDerly on your own data, and want to specify the input and output path:

```orderly extract --input_path="/data/ord/" --output_path="/data/orderly/"```

This will generate a parquet file for each ORD file.

### Cleaning the data

```orderly clean```

This will produce train and test parquet files, along with a .json file showing the arguments used and a .log file showing the operations run.

## Training a condition prediction algorithm with this data
 @Kobi
For this, clone the repository and use the makefile.

### Requirements
Python dependencies can be installed via ```poetry``` from within the ```orderly/condition_prediction``` folder:

- run in terminal: ```poetry install```

### Train model



@Kobi see inspiration below:
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
@Kobi see inspiration above




# Customisable cleaning of ORD data



# Customisable setup

## Installation and download
Same as above

## Extraction
There are two different ways to extract data from ORD files, trusting the labelling, or using the reaction string (as specified in the ```trust_labelling``` boolean). Below you see all the arguments that can be passed to the extraction script, change as appropriate:

``` orderly extract --name_contains_substring="uspto" --trust_labelling=False --output_path="data/orderly/uspto_no_trust" --consider_molecule_names=False```

## Cleaning
There are also a number of customisable steps for the cleaning:

```orderly clean --output_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=True --scramble=True --train_test_split_fraction=0.9```



# Reproducing results from paper

To reproduce the results from the paper, please clone the repository, and use poetry to install the requirements (see above). Towards the bottom of the makefile, you will find a comprehensive 8 step list of steps to generate all the datasets and reproduce all results presented in the paper. 



# Results

We run the condition prediction model on four different datasets, and find that trusting the labelling of the ORD data leads to overly confident test accuracy. We conclude that applying chemical logic to the reaction string is necessary to get a high-quality dataset, and that the best strategy for dealing with rare molecules is to delete reactions where they appear.

Top-3 exact match combination accuracy (\%): frequency informed guess  // model prediction  //  AIB\%:

| Dataset            | A (labeling; rare->"other")   | B (labeling; rare->delete rxn) | C (reaction string; rare->"other") | D (reaction string; rare->delete rxn) |
|--------------------|--------------------------------|---------------------------------|------------------------------------|--------------------------------------|
| Solvents           | 47 // 58 // 21%                | 50 // 61 // 22%                 | 23 // 42 // 26%                    | 24 // 45 // 28%                      |
| Agents             | 54 // 70 // 35%                | 58 // 72 // 32%                 | 19 // 39 // 25%                    | 21 // 42 // 27%                      |
| Solvents & Agents  | 31 // 44 // 19%                | 33 // 47 // 21%                 | 4 // 21 // 18%                     | 5 // 24 // 21%                       |





## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
























### 2. Run extraction

We can run extraction using: ```poetry run python -m orderly.extract```. Using ```poetry run python -m orderly.extract --help``` will explain the arguments. Certain args must be set such as data paths.

### 3. Run cleaning

We can run cleaning using: ```poetry run python -m orderly.clean```. Using ```poetry run python -m orderly.clean --help``` will explain the arguments. Certain args must be set such as data paths.

