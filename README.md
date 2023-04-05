# ORDerly

A wrapper for downloading ORD-schema data, extracting and cleaning the data

The scripts herein will extract and clean data from ORD with various manual steps relying on chemical domain knowledge. This results in an open-source dataset containing a mapped reaction, reactants, products, solvents, reagents, catalysts, and yields in a pandas DataFrame structure that should also be easily usable by people with little knowledge of chemistry.

# Usage

### 1. Install

#### I Download the ORD data

We want to download the ORD data locally, this can be done through any of the following methods:

1. Follow the instructions at: https://github.com/open-reaction-database/ord-data, we specifically care about the folders in ```ord-data/data/```
2. Docker install with linux (run in terminal):
    ```
    make linux_download_ord
    ``` 
3. Docker install with mac (run in terminal):
    ```
    make root_download_ord
    make sudo_chown
    ```

#### II Install OS depenencies
 


#### III Install dependencies

To install the dependencies this can be done via ```poetry``` or you can run the environment through docker.

1. For poetry (run in terminal):
    You might need some environment dependencies
        Linux: For you will likely have some missing dependencies, these can be installed via apt for example: 
        ```
        sudo apt-get update
        sudo apt-get install libpq-dev gcc -y
        ```
    Python dependencies:
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


## Appendix

### Solvents

In data/solvents.csv you'll find a list of solvens which we use to label solvents (to avoid relying on the labelling in USPTO), this list was created from the intersection of the following two lists (excluding acids, bases, and polymers):
 - https://github.com/sustainable-processes/vle_prediction/blob/master/data/cosmo/solvent_descriptors.csv
 - https://github.com/sustainable-processes/summit/blob/main/data/ucb_pharma_approved_list.csv
