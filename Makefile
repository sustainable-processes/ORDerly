current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)
download_path=ord/

clean_default_num_agent=3
clean_default_num_cat=1
clean_default_num_reag=2
WANDB_ENTITY=ceb-sre
dataset_version=v6


mypy_orderly:
	python -m mypy . --ignore-missing-imports --explicit-package-bases --exclude condition_prediction

strict_mypy_orderly:
	python -m mypy . --ignore-missing-imports --exclude condition_prediction --explicit-package-bases --strict

mypy_condition_prediction:
	echo "Make sure to run this in the condition_prediction environment"
	python -m mypy condition_prediction --ignore-missing-imports --explicit-package-bases

black:
	python -m black .

test_extract:
	python -m pytest -vv tests/test_extract.py

test_clean:
	python -m pytest -vv tests/test_clean.py

test_data:
	python -m pytest -vv tests/test_data.py

pytest_orderly:
	python -m pytest -vv tests/

pytest_condition_prediction:
	echo "Make sure to run this in the condition_prediction environment"
	python -m pytest -vv condition_prediction/

pytestx:
	python -m pytest -vv --exitfirst

gen_test_data:
	python -m orderly.extract --data_path=orderly/data/test_data/ord_test_data --output_path=orderly/data/test_data/extracted_ord_test_data_trust_labelling  --trust_labelling=True --name_contains_substring="" --overwrite=False --use_multiprocessing=True
	python -m orderly.extract --data_path=orderly/data/test_data/ord_test_data --output_path=orderly/data/test_data/extracted_ord_test_data_dont_trust_labelling  --trust_labelling=False --name_contains_substring="" --overwrite=False --use_multiprocessing=True

build_orderly:
	docker image build --target orderly_base --tag orderly_base .
	docker image build --target orderly_base_sudo --tag orderly_base_sudo .

run_orderly:
	docker run -v $(current_dir)/data:/home/worker/repo/data/ -u $(uid):$(gid) -it orderly_base

run_orderly_sudo:
	docker run -v $(current_dir)/data:/home/worker/repo/data/ -it orderly_base_sudo

build_orderly_from_pip:
	docker image build --target orderly_pip --tag orderly_pip .

run_orderly_from_pip:
	docker run -v $(current_dir)/data:/home/worker/repo/data/ -u $(uid):$(gid) -it orderly_pip

run_orderly_black:
	docker image build --target orderly_black --tag orderly_black .
	docker run -v $(current_dir):/home/worker/repo/ -u $(uid):$(gid) orderly_black

run_orderly_pytest:
	docker image build --target orderly_test --tag orderly_test .
	docker run orderly_test

run_orderly_mypy:
	docker image build --target orderly_mypy --tag orderly_mypy .
	docker run orderly_mypy

run_orderly_mypy_strict:
	docker image build --target orderly_mypy_strict --tag orderly_mypy_strict .
	docker run orderly_mypy_strict

run_orderly_gen_test_data:
	docker image build --target orderly_gen_test_data --tag orderly_gen_test_data .
	docker run -v $(current_dir)/orderly/data/:/home/worker/repo/orderly/data/ -u $(uid):$(gid) orderly_gen_test_data

linux_download_ord:
	docker image build --target orderly_download_linux --tag orderly_download_linux .
	docker run -v $(current_dir)/data:/tmp_data -u $(uid):$(gid) orderly_download_linux

_linux_get_ord:
	mkdir -p /tmp_data/${download_path}
	touch /tmp_data/${download_path}/tst_permissions_file.txt
	rm /tmp_data/${download_path}/tst_permissions_file.txt
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /app
	cp -a /app/ord-data-main/data/. /tmp_data/${download_path}

root_download_ord:
	docker image build --target orderly_download_root --tag orderly_download_root .
	docker run -v $(current_dir)/data:/tmp_data orderly_download_root
	
_root_get_ord:
	mkdir -p /tmp_data/${download_path}
	touch /tmp_data/${download_path}/tst_permissions_file.txt
	rm /tmp_data/${download_path}/tst_permissions_file.txt
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /app
	cp -a /app/ord-data-main/data/. /tmp_data/${download_path}

sudo_chown:
	sudo chown -R $(uid):$(gid) $(current_dir)

run_python_310:
	docker run -it python:3.10-slim-buster /bin/bash


####################################################################################################
# 									ORDerly make commands for the paper
####################################################################################################

### Steps:
# 1. Extract uspto data, trust_labelling = False
# 2. Clean with set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True, remove reactions with no reactants or products, consistent_yield=False, no filtering
# 3. Plot histograms of the number of non-empty columns of each type (reactants, products, solvents, agents)
# 4. Run a cleaning with decided upon number of columns to keep
# 5. Plot histogram showing dataset size as a function of min_frequency_of_occurrence (can probably use the min_frequency code from the cleaner within the plotter)
# 6. Generate the four datasets we need for the paper (split into train and test set)
# 7. Plot histograms with the occurrence of the most common reactants, products, solvents, agents
# 8. Generate fingerprints for each dataset
# 9. Train & evaluate a model on each dataset

# 1. Extract 

paper_extract_uspto_no_trust:
	python -m orderly.extract --name_contains_substring="uspto" --trust_labelling=False --output_path="data/orderly/uspto_no_trust" --consider_molecule_names=False

paper_extract_uspto_with_trust:
	python -m orderly.extract --name_contains_substring="uspto" --trust_labelling=True --output_path="data/orderly/uspto_with_trust" --consider_molecule_names=True

paper_1: paper_extract_uspto_no_trust paper_extract_uspto_with_trust

# 2. Clean (unfiltered)

paper_clean_uspto_no_trust_unfiltered: #requires: paper_extract_uspto_no_trust
	python -m orderly.clean --output_path="data/orderly/uspto_no_trust/unfiltered/unfiltered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=-1 --num_reactant=-1 --num_solv=-1 --num_agent=-1 --num_cat=0 --num_reag=0 --consistent_yield=False --train_size=0.0

paper_clean_uspto_with_trust_unfiltered: #requires: paper_extract_uspto_with_trust
	python -m orderly.clean --output_path="data/orderly/uspto_with_trust/unfiltered/unfiltered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=-1 --num_reactant=-1 --num_solv=-1 --num_agent=0 --num_cat=-1 --num_reag=-1 --consistent_yield=False --train_size=0.0

paper_2: paper_clean_uspto_no_trust_unfiltered paper_clean_uspto_with_trust_unfiltered

# 3. Plots

paper_plot_uspto_no_trust_unfiltered_num_rxn_components: #requires: paper_clean_uspto_no_trust_unfiltered
	python -m orderly.plot --clean_data_path="data/orderly/uspto_no_trust/unfiltered/unfiltered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_no_trust/" --plot_num_rxn_components_bool=True --plot_frequency_of_occurrence_bool=False --plot_molecule_popularity_histograms=False

paper_plot_uspto_with_trust_unfiltered_num_rxn_components: #requires: paper_clean_uspto_with_trust_unfiltered
	python -m orderly.plot --clean_data_path="data/orderly/uspto_with_trust/unfiltered/unfiltered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=True --plot_frequency_of_occurrence_bool=False --plot_molecule_popularity_histograms=False

paper_3: paper_plot_uspto_no_trust_unfiltered_num_rxn_components paper_plot_uspto_with_trust_unfiltered_num_rxn_components

# 4. clean (filtered)

paper_clean_uspto_no_trust_filtered: #requires: paper_extract_uspto_no_trust
	python -m orderly.clean --output_path="data/orderly/uspto_no_trust/filtered/filtered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=False --remove_reactions_with_no_solvents=False --remove_reactions_with_no_agents=False --train_size=0.0

paper_clean_uspto_with_trust_filtered: #requires: paper_extract_uspto_with_trust
	python -m orderly.clean --output_path="data/orderly/uspto_with_trust/filtered/filtered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=0 --num_cat=1 --num_reag=2 --consistent_yield=False --remove_reactions_with_no_solvents=False --remove_reactions_with_no_agents=False --train_size=0.0

paper_4: paper_clean_uspto_no_trust_filtered paper_clean_uspto_with_trust_filtered

# 5. plot min freq of occurence
paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_10_100:
	python -m orderly.plot --clean_data_path="data/orderly/uspto_no_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_no_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_molecule_popularity_histograms=False --freq_threshold=100 --freq_step=10

paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_100_1000:
	python -m orderly.plot --clean_data_path="data/orderly/uspto_no_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_no_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_molecule_popularity_histograms=False --freq_threshold=1000 --freq_step=100
	

paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_10_100:
	python -m orderly.plot --clean_data_path="data/orderly/uspto_with_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_molecule_popularity_histograms=False --freq_threshold=100 --freq_step=10

paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_100_1000:
	python -m orderly.plot --clean_data_path="data/orderly/uspto_with_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_molecule_popularity_histograms=False --freq_threshold=1000 --freq_step=100

paper_5 : paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_10_100 paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_100_1000 paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_10_100 paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_100_1000


# 6. clean (final)
# NB: I changed this one, min_freq=0, train_size=1
paper_gen_orderly_cond_prelim_100: #requires: paper_extract_uspto_no_trust
	python -m orderly.clean --output_path="data/orderly/datasets_$(dataset_version)/orderly_cond_prelim_100.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=False --scramble=True --train_size=1

paper_gen_uspto_no_trust_no_map: #requires: paper_extract_uspto_no_trust
	python -m orderly.clean --output_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=False --scramble=True --train_size=0.9

paper_gen_uspto_no_trust_with_map: #requires: paper_extract_uspto_no_trust
	python -m orderly.clean --output_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=False --scramble=True --train_size=0.9

paper_gen_uspto_with_trust_with_map: #requires: paper_extract_uspto_with_trust
	python -m orderly.clean --output_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=0 --num_cat=1 --num_reag=2 --consistent_yield=False --scramble=True --train_size=0.9

paper_gen_uspto_with_trust_no_map: #requires: paper_extract_uspto_with_trust
	python -m orderly.clean --output_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=0 --num_cat=1 --num_reag=2 --consistent_yield=False --scramble=True --train_size=0.9

paper_6: paper_gen_uspto_no_trust_no_map paper_gen_uspto_no_trust_with_map paper_gen_uspto_with_trust_with_map paper_gen_uspto_with_trust_no_map

# 7. Plot plot_molecule_popularity_histograms 
paper_plot_uspto_no_trust_no_map:
	python -m orderly.plot --clean_data_path="data/orderly/datasets/orderly_no_trust_no_map_train.parquet" --plot_output_path="data/orderly/plot_no_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=False --plot_molecule_popularity_histograms=True 

paper_plot_uspto_with_trust_no_map:
	python -m orderly.plot --clean_data_path="data/orderly/datasets/orderly_with_trust_no_map_train.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=False --plot_molecule_popularity_histograms=True
	
paper_7 : paper_plot_uspto_no_trust_no_map  paper_plot_uspto_with_trust_no_map

# 8. gen fp

fp_no_trust_no_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_test.parquet" --fp_size=2048 --overwrite=False
fp_no_trust_no_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_train.parquet" --fp_size=2048 --overwrite=False

fp_no_trust_with_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_test.parquet" --fp_size=2048 --overwrite=False
fp_no_trust_with_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_train.parquet" --fp_size=2048 --overwrite=False

fp_with_trust_with_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_test.parquet" --fp_size=2048 --overwrite=False
fp_with_trust_with_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_train.parquet" --fp_size=2048 --overwrite=False

fp_with_trust_no_map_test:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_test.parquet" --fp_size=2048 --overwrite=False
fp_with_trust_no_map_train:
	python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_train.parquet" --fp_size=2048 --overwrite=False

paper_8: fp_no_trust_no_map_test fp_no_trust_no_map_train fp_no_trust_with_map_test fp_no_trust_with_map_train fp_with_trust_with_map_test fp_with_trust_with_map_train fp_with_trust_no_map_test fp_with_trust_no_map_train

#Generate datasets for paper
paper_get_datasets: paper_1 paper_6 paper_8

paper_gen_all: paper_1 paper_2 paper_3 paper_4 paper_5 paper_6 paper_8

# 9. train models
#Remember to switch env here (must contain TF, e.g. tf_mac_m1)
# Full dataset
no_trust_no_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_test.parquet" --output_folder_path="models/no_trust_no_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

no_trust_with_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_no_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_test.parquet" --output_folder_path="models/with_trust_no_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_with_map_train:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_test.parquet" --output_folder_path="models/with_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

# 20% of data
no_trust_no_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_no_map_test.parquet" --output_folder_path="models/no_trust_no_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

no_trust_with_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_no_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_no_map_test.parquet" --output_folder_path="models/with_trust_no_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY) --dataset_version=$(datset_version)

with_trust_with_map_train_20:
	python -m condition_prediction --train_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets_$(dataset_version)/orderly_with_trust_with_map_test.parquet" --output_folder_path="models/with_trust_with_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=$(WANDB_ENTITY)


# Sweeps
RANDOM_SEEDS = 12345 54321 98765
TRAIN_FRACS =   1.0 0.2 0.4 0.6 0.8
# Path on lightning
# DATASETS_PATH = /project/studios/orderly-preprocessing/ORDerly/data/orderly/datasets_$(dataset_version)/
# Normal path
DATASETS_PATH = ORDerly/data/orderly/datasets_$(dataset_version)/
DATASETS = no_trust_with_map  no_trust_no_map with_trust_with_map with_trust_no_map 
dataset_size_sweep:
	@for random_seed in ${RANDOM_SEEDS}; \
	do \
		for dataset in ${DATASETS}; \
		do \
			for train_frac in ${TRAIN_FRACS}; \
			do \
				rm -rf .tf_cache* && python -m condition_prediction --train_data_path=${DATASETS_PATH}/orderly_$${dataset}_train.parquet --test_data_path=${DATASETS_PATH}/orderly_$${dataset}_test.parquet --output_folder_path=models/$${dataset} --dataset_version=$(datset_version) --train_fraction=$${train_frac} --train_val_split=0.8 --random_seed=$${random_seed} --overwrite=True --batch_size=512 --epochs=100 --train_mode=0 --early_stopping_patience=0  --evaluate_on_test_data=True --wandb_entity=$(WANDB_ENTITY) ; \
			done \
		done \
	done


sweep_no_trust_no_map_train_commands:
	python -m sweep sweeps/no_trust_no_map_train.yaml --dry_run

sweep_no_trust_with_map_train_commands:
	python -m sweep sweeps/no_trust_with_map_train.yaml --dry_run

sweep_with_trust_no_map_train_commands:
	python -m sweep sweeps/with_trust_no_map_train.yaml --dry_run

sweep_with_trust_with_map_train_commands:
	python -m sweep sweeps/with_trust_with_map_train.yaml --dry_run

sweep_all: sweep_no_trust_no_map_train_commands sweep_no_trust_with_map_train_commands sweep_with_trust_no_map_train_commands sweep_with_trust_with_map_train_commands

train_all: no_trust_no_map_train no_trust_with_map_train with_trust_no_map_train with_trust_with_map_train no_trust_no_map_train_20 no_trust_with_map_train_20 with_trust_no_map_train_20 with_trust_with_map_train_20

# Example workflow for running everything: no trust, with map
# extract, clean, fp
example_workflow_prep_data: paper_extract_uspto_no_trust paper_gen_uspto_no_trust_with_map fp_no_trust_with_map_test fp_no_trust_with_map_train

# change env to have TF
example_workflow_train_model: no_trust_with_map_train

