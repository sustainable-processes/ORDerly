# ORDerly

Cleaning and extraction of data from ORD

The scripts herein will extract and clean data from ORD with various manual steps relying on chemical domain knowledge. This results in an open-source dataset containing a mapped reaction, reactants, products, solvents, reagents, catalysts, and yields in a pandas DataFrame structure that should also be easily usable by people with little knowledge of chemistry.

# Usage

### 1. Install

#### I Download the ORD data

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

#### II Install OS depenencies
 
You might need some environment dependencies. If running locally these will need to be dealt with. However, if running using docker, the depenencies will be managed in the build script.

- Linux: For you will likely have some missing dependencies, these can be installed via apt for example: 

```
sudo apt-get update
sudo apt-get install libpq-dev gcc -y
```

#### III Install Python dependencies

To install the dependencies this can be done via ```poetry``` or you can run the environment through docker.

- For poetry (run in terminal):
    Python dependencies: ```poetry install```
- For docker (run in terminal):
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

# ML models trained on ORDerly

We plan to show the usefulness of ORDerly by training ML models from the literature on ORDerly for standardised prediction tasks. Prediction tasks include:
- Yield prediction
    - https://chemrxiv.org/engage/chemrxiv/article-details/6150143118be8575b030ad43
- Retrosynthesis
- Forward prediction
- Condition prediction

We may be able to use https://deepchem.io/models


## Appendix

### Solvents

In data/solvents.csv you'll find a list of solvens which we use to label solvents (to avoid relying on the labelling in ORD), this list was created from the intersection of solvents coming from three different sources. The following procedure was followed for the construction of solvents.csv:

1. Data curation: We compiled a list of solvent names from the following 3 sources. Unfortunately they did not include SMILES strings.
 - https://doi.org/10.1039/C9SC01844A
 - https://www.acs.org/greenchemistry/research-innovation/tools-for-green-chemistry/solvent-selection-tool.html
 - https://github.com/sustainable-processes/summit/blob/main/data/ucb_pharma_approved_list.csv

2. Filtering: Make all solvent names lower case, strip spaces, find and remove duplicate names. (Before: 458+272+115=845 rows in total. After removing duplicates: 615)
3. Name resolution: The dataframe has 4 columns of identifiers: 3 for (english) solvent names, and 1 for CAS numbers. We ran (Pura)[https://github.com/sustainable-processes/pura] with <services=[PubChem(autocomplete=True), Opsin(), CIR(),]> and <agreement=2> separately on each of the three solvent name columns, and <services=[CAS()]>, <agreement=1> to resolve the CAS numbers.
4. Agreement: We now had up to 4 SMILES strings for each solvent, and the SMILES string was trusted when all of them were in agreement (either the other items were the same SMILES or empty). There were ~40 rows with disagreement, and these were resolved manually (by cross-checking name and CAS with PubChem/Wikipedia).
5. The final dataset consists of the following columns: 'solvent_name_1', 'solvent_name_2', 'solvent_name_3', 'cas_number', 'chemical_formula', 'smiles', 'source'.
