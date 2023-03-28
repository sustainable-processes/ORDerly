# USPTO_cleaning
Cleaning and extraction of USPTO data from ORD

The scripts herein will extract and clean USPTO data from ORD with various manual steps relying on chemical domain knowledge. This results in an open-source dataset containing a mapped reaction, reactants, products, solvents, reagents, catalysts, and yields in a pandas DataFrame structure that should also be easily usable by people with little knowledge of chemistry.

# Instructions
1. Git clone this repo
2. Download the USPTO data from ORD into data/USPTO/:
    - While inside USPTO: 
```    
git clone https://github.com/open-reaction-database/ord-data
```
    - You'll find the data in ord-data/data/
    - You'll notice that the data is split into folders, each containing a number of ORD files. They are batched by year.
3. Run the following command in the root directory (This took 19 min on a mac studio):
```
python USPTO_extraction.py True
```
    - For alternate usage, see the documentation in the corresponsing file
4. Run the following command in the root director (This took 8 min on a mac studio):

```
python USPTO_cleaning.py --clean_data_file_name=cleaned_USPTO --consistent_yield=True --num_reactant=5 --num_product=5 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --min_frequency_of_occurance=100
```

For alternate usage, see the documentation in the corresponsing file
5. Find the cleaned USPTO data in the data folder 



For ubuntu: 
```
sudo apt-get install libpq-dev
```


## Solvents
In data/solvents.csv you'll find a list of solvens which we use to label solvents (to avoid relying on the labelling in USPTO), this list was created from the intersection of the following two lists (excluding acids, bases, and polymers):
 - https://github.com/sustainable-processes/vle_prediction/blob/master/data/cosmo/solvent_descriptors.csv
 - https://github.com/sustainable-processes/summit/blob/main/data/ucb_pharma_approved_list.csv


make build_download_ord

make run_download_ord

poetry install

poetry run python -m orderly.extract [--help]

poetry run python -m orderly.clean [--help]


# ORDerly

A wrapper for downloading ORDschema data, extracting and cleaning the data

The scripts herein will extract and clean USPTO data from ORD with various manual steps relying on chemical domain knowledge. This results in an open-source dataset containing a mapped reaction, reactants, products, solvents, reagents, catalysts, and yields in a pandas DataFrame structure that should also be easily usable by people with little knowledge of chemistry.

## Install

### 1. Download the ORD data

We want to download the ORD data locally, this can be done through any of the following methods:

1. Follow the instructions at: https://github.com/open-reaction-database/ord-data, we specifically care about the folders in ```ord-data/data/`
2. Docker install with linux (run in terminal):
    ```
    make linux_download_ord
    ``` 
3. Docker install with mac (run in terminal):
    ```
    make root_download_ord
    make sudo_chown
    ```

### 3. Install dependencies

To install the dependencies this can be done via ```poetry``` or you can run the environment through docker.

1. For poetry (run in terminal):
    ```bash
    poetry install
    ```
2. For docker (run in terminal):
    ```bash
    build_orderly
    run_orderly
    ```
    You can validate the install works by running
    ```bash
    build_orderly_extras
    run_orderly_pytest
    ```


### 2. Run extraction

We can run extraction using: ```poetry run python -m orderly.extract```. Using ```poetry run python -m orderly.extract --help``` will explain the arguments. Certain args must be set such as data paths.

### 3. Run cleaning

We can run cleaning using: ```poetry run python -m orderly.clean```. Using ```poetry run python -m orderly.clean --help``` will explain the arguments. Certain args must be set such as data paths.
