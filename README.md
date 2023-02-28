# USPTO_cleaning
Cleaning and extraction of USPTO data from ORD

The scripts herein will extract and clean USPTO data from ORD with various manual steps relying on chemical domain knowledge. This results in an open-source dataset containing a mapped reaction, reactants, products, solvents, reagents, catalysts, and yields in a pandas DataFrame structure that should also be easily usable by people with little knowledge of chemistry.


## Solvents
In data/solvents.csv you'll find a list of solvens which we use to label solvents (to avoid relying on the labelling in USPTO), this list was created from the intersection of the following two lists (excluding acids, bases, and polymers):
 - https://github.com/sustainable-processes/vle_prediction/blob/master/data/cosmo/solvent_descriptors.csv
 - https://github.com/sustainable-processes/summit/blob/main/data/ucb_pharma_approved_list.csv
 