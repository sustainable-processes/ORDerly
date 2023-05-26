current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)
download_path=ord/

clean_default_num_agent=3
clean_default_num_cat=1
clean_default_num_reag=2
WANDB_ENTITY=ceb-sre


mypy:
	poetry run python -m mypy . --ignore-missing-imports

strict_mypy:
	poetry run python -m mypy . --ignore-missing-imports --strict

black:
	poetry run python -m black .

test_extract:
	poetry run python -m pytest -vv tests/test_extract.py

test_clean:
	poetry run python -m pytest -vv tests/test_clean.py

test_data:
	poetry run python -m pytest -vv tests/test_data.py

pytest:
	poetry run python -m pytest -vv

pytestx:
	poetry run python -m pytest -vv --exitfirst

extract_all_no_trust:
	poetry run python -m orderly.extract --name_contains_substring="" --trust_labelling=False --output_path="data/orderly/all_no_trust"

clean_all_no_trust:
	poetry run python -m orderly.clean --output_path="data/orderly/all_no_trust/orderly_ord.parquet" --ord_extraction_path="data/orderly/all_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/all_no_trust/all_molecule_names.csv" --num_agent=$(clean_default_num_agent) --num_cat=0 --num_reag=0

gen_all_no_trust: extract_all_no_trust clean_all_no_trust
	
extract_all_trust:
	poetry run python -m orderly.extract --name_contains_substring="" --trust_labelling=True --output_path="data/orderly/all_trust"

clean_all_trust:
	poetry run python -m orderly.clean --output_path="data/orderly/all_trust/orderly_ord.parquet" --ord_extraction_path="data/orderly/all_trust/extracted_ords" --molecules_to_remove_path="data/orderly/all_trust/all_molecule_names.csv" --num_agent=0 --num_cat=$(clean_default_num_cat) --num_reag=$(clean_default_num_reag)

gen_all_trust: extract_all_trust clean_all_trust
	
extract_uspto_no_trust:
	poetry run python -m orderly.extract --name_contains_substring="uspto" --trust_labelling=False --output_path="data/orderly/uspto_no_trust"

clean_uspto_no_trust:
	poetry run python -m orderly.clean --output_path="data/orderly/uspto_no_trust/orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --num_agent=$(clean_default_num_agent) --num_cat=0 --num_reag=0

gen_uspto_no_trust: extract_uspto_no_trust clean_uspto_no_trust
	
extract_uspto_trust:
	poetry run python -m orderly.extract --name_contains_substring="uspto" --trust_labelling=True --output_path="data/orderly/uspto_trust"

clean_uspto_trust:
	poetry run python -m orderly.clean --output_path="data/orderly/uspto_trust/orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_trust/all_molecule_names.csv" --num_agent=0 --num_cat=$(clean_default_num_cat) --num_reag=$(clean_default_num_reag) 

gen_uspto_trust: extract_uspto_trust clean_uspto_trust
	
gen_datasets: gen_uspto_no_trust gen_uspto_trust gen_all_no_trust gen_all_trust

clean_unfiltered_uspto_no_trust:
	poetry run python -m orderly.clean --output_path="data/orderly/unfiltered/uspto_no_trust/orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=100 --num_reactant=100 --num_solv=100 --num_agent=100 --num_cat=0 --num_reag=0 --consistent_yield=False	

clean_unfiltered_uspto_trust:
	poetry run python -m orderly.clean --output_path="data/orderly/unfiltered/uspto_trust/orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=100 --num_reactant=100 --num_solv=100 --num_agent=0 --num_cat=100 --num_reag=100 --consistent_yield=False

clean_unfiltered_uspto: clean_unfiltered_uspto_no_trust clean_unfiltered_uspto_trust

gen_unfiltered_uspto: extract_uspto_no_trust extract_uspto_trust clean_unfiltered_uspto 

scramble_true:
	poetry run python -m orderly.clean --output_path="data/orderly/check_scramble/orderly_no_trust_no_map_with_scramble.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=1000 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=True --scramble=True --apply_random_split=False

scramble_false:
	poetry run python -m orderly.clean --output_path="data/orderly/check_scramble/orderly_no_trust_no_map_no_scramble.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=1000 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=True --scramble=False --apply_random_split=False

gen_test_data:
	poetry run python -m orderly.extract --data_path=orderly/data/test_data/ord_test_data --output_path=orderly/data/test_data/extracted_ord_test_data_trust_labelling  --trust_labelling=True --name_contains_substring="" --overwrite=False --use_multiprocessing=True
	poetry run python -m orderly.extract --data_path=orderly/data/test_data/ord_test_data --output_path=orderly/data/test_data/extracted_ord_test_data_dont_trust_labelling  --trust_labelling=False --name_contains_substring="" --overwrite=False --use_multiprocessing=True

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

build_rxnmapper:
	docker image build --target rxnmapper_base --tag rxnmapper_base .

run_rxnmapper:
	docker run -v $(current_dir)/data:/tmp_data -it rxnmapper_base

run_python_310:
	docker run -it python:3.10-slim-buster /bin/bash



train_model:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=True --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5




####################################################################################################
# 									ORDerly make commands for the paper
####################################################################################################

### Steps:
# 1. Extract uspto data, trust_labelling = False
# 2. Clean with set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True, remove reactions with no reactants or products, consistent_yield=True, no filtering
# 3. Plot histograms of the number of non-empty columns of each type (reactants, products, solvents, agents)
# 4. Run a cleaning with decided upon number of columns to keep
# 5. Plot histogram showing dataset size as a function of min_frequency_of_occurrence (can probably use the min_frequency code from the cleaner within the plotter)
# 6. Generate the four datasets we need for the paper (split into train and test set)
# 7. Generate fingerprints for each dataset
# 8. Train & evaluate a model on each dataset


paper_extract_uspto_no_trust:
	poetry run python -m orderly.extract --name_contains_substring="uspto" --trust_labelling=False --output_path="data/orderly/uspto_no_trust"

paper_extract_uspto_with_trust:
	poetry run python -m orderly.extract --name_contains_substring="uspto" --trust_labelling=True --output_path="data/orderly/uspto_with_trust"

paper_1: paper_extract_uspto_no_trust paper_extract_uspto_with_trust

# 2.

paper_clean_uspto_no_trust_unfiltered:
	poetry run python -m orderly.clean --output_path="data/orderly/uspto_no_trust/unfiltered/unfiltered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=20 --num_reactant=20 --num_solv=20 --num_agent=20 --num_cat=0 --num_reag=0 --consistent_yield=False

paper_clean_uspto_with_trust_unfiltered:
	poetry run python -m orderly.clean --output_path="data/orderly/uspto_with_trust/unfiltered/unfiltered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=20 --num_reactant=20 --num_solv=20 --num_agent=0 --num_cat=10 --num_reag=10 --consistent_yield=False

paper_2: paper_clean_uspto_no_trust_unfiltered paper_clean_uspto_with_trust_unfiltered
# 3.

paper_plot_uspto_no_trust_unfiltered_num_rxn_components:
	poetry run python -m orderly.plot --clean_data_path="data/orderly/uspto_no_trust/unfiltered/unfiltered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_no_trust/" --plot_num_rxn_components_bool=True --plot_frequency_of_occurrence_bool=False --plot_waterfall_bool=False

paper_plot_uspto_with_trust_unfiltered_num_rxn_components:
	poetry run python -m orderly.plot --clean_data_path="data/orderly/uspto_with_trust/unfiltered/unfiltered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=True --plot_frequency_of_occurrence_bool=False --plot_waterfall_bool=False

paper_3: paper_plot_uspto_no_trust_unfiltered_num_rxn_components paper_plot_uspto_with_trust_unfiltered_num_rxn_components

# 4.

paper_clean_uspto_no_trust_filtered:
	poetry run python -m orderly.clean --output_path="data/orderly/uspto_no_trust/filtered/filtered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=False

paper_clean_uspto_with_trust_filtered:
	poetry run python -m orderly.clean --output_path="data/orderly/uspto_with_trust/filtered/filtered_orderly_ord.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=0 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=0 --num_cat=1 --num_reag=2 --consistent_yield=False

paper_4: paper_clean_uspto_no_trust_filtered

# 5.
paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_10_100:
	poetry run python -m orderly.plot --clean_data_path="data/orderly/uspto_no_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_waterfall_bool=False --freq_threshold=100 --freq_step=10

paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_100_1000:
	poetry run python -m orderly.plot --clean_data_path="data/orderly/uspto_no_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_waterfall_bool=False --freq_threshold=1000 --freq_step=100

paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_10_100:
	poetry run python -m orderly.plot --clean_data_path="data/orderly/uspto_with_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_waterfall_bool=False --freq_threshold=100 --freq_step=10

paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_100_1000:
	poetry run python -m orderly.plot --clean_data_path="data/orderly/uspto_with_trust/filtered/filtered_orderly_ord.parquet" --plot_output_path="data/orderly/plot_with_trust/" --plot_num_rxn_components_bool=False --plot_frequency_of_occurrence_bool=True --plot_waterfall_bool=False --freq_threshold=1000 --freq_step=100

paper_5 : paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_10_100 paper_plot_uspto_no_trust_filtered_min_frequency_of_occurrence_100_1000 paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_10_100 paper_plot_uspto_with_trust_filtered_min_frequency_of_occurrence_100_1000


# 6.
paper_gen_uspto_no_trust_no_map:
	poetry run python -m orderly.clean --output_path="data/orderly/datasets/orderly_no_trust_no_map.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=True --scramble=True --train_test_split_fration=0.9

paper_gen_uspto_no_trust_with_map:
	poetry run python -m orderly.clean --output_path="data/orderly/datasets/orderly_no_trust_with_map.parquet" --ord_extraction_path="data/orderly/uspto_no_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_no_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --consistent_yield=True --scramble=True --train_test_split_fration=0.9

paper_gen_uspto_with_trust_with_map:
	poetry run python -m orderly.clean --output_path="data/orderly/datasets/orderly_with_trust_with_map.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=True --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=0 --num_cat=1 --num_reag=2 --consistent_yield=True --scramble=True --train_test_split_fration=0.9

paper_gen_uspto_with_trust_no_map:
	poetry run python -m orderly.clean --output_path="data/orderly/datasets/orderly_with_trust_no_map.parquet" --ord_extraction_path="data/orderly/uspto_with_trust/extracted_ords" --molecules_to_remove_path="data/orderly/uspto_with_trust/all_molecule_names.csv" --min_frequency_of_occurrence=100 --map_rare_molecules_to_other=False --set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=True --remove_rxn_with_unresolved_names=False --set_unresolved_names_to_none=False --num_product=1 --num_reactant=2 --num_solv=2 --num_agent=0 --num_cat=1 --num_reag=2 --consistent_yield=True --scramble=True --train_test_split_fration=0.9

paper_6: paper_gen_uspto_no_trust_no_map paper_gen_uspto_no_trust_with_map paper_gen_uspto_with_trust_with_map paper_gen_uspto_with_trust_no_map

# 7.

fp_no_trust_no_map_test:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_no_trust_no_map_test.parquet" --fp_size=2048 --overwrite=False
fp_no_trust_no_map_train:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_no_trust_no_map_train.parquet" --fp_size=2048 --overwrite=False

fp_no_trust_with_map_test:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_no_trust_with_map_test.parquet" --fp_size=2048 --overwrite=False
fp_no_trust_with_map_train:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_no_trust_with_map_train.parquet" --fp_size=2048 --overwrite=False

fp_with_trust_with_map_test:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_with_trust_with_map_test.parquet" --fp_size=2048 --overwrite=False
fp_with_trust_with_map_train:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_with_trust_with_map_train.parquet" --fp_size=2048 --overwrite=False

fp_with_trust_no_map_test:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_with_trust_no_map_test.parquet" --fp_size=2048 --overwrite=False
fp_with_trust_no_map_train:
	poetry run python -m orderly.gen_fp --clean_data_folder_path="data/orderly/datasets/orderly_with_trust_no_map_train.parquet" --fp_size=2048 --overwrite=False

paper_7: fp_no_trust_no_map_test fp_no_trust_no_map_train fp_no_trust_with_map_test fp_no_trust_with_map_train fp_with_trust_with_map_test fp_with_trust_with_map_train fp_with_trust_no_map_test fp_with_trust_no_map_train

#Generate datasets for paper
paper_get_datasets: paper_1 paper_6 paper_7

paper_gen_all: paper_1 paper_2 paper_3 paper_4 paper_5 paper_6 paper_7

# 8.
#Remember to switch env here (must contain TF, e.g. tf_mac_m1)
# Full dataset
no_trust_no_map_train:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_no_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_no_trust_no_map_test.parquet" --output_folder_path="models/no_trust_no_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

no_trust_with_map_train:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

with_trust_no_map_train:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_with_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_with_trust_no_map_test.parquet" --output_folder_path="models/with_trust_no_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

with_trust_with_map_train:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_with_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_with_trust_with_map_test.parquet" --output_folder_path="models/with_trust_with_map"  --train_fraction=1 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

# 20% of data
no_trust_no_map_train_20:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_no_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_no_trust_no_map_test.parquet" --output_folder_path="models/no_trust_no_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

no_trust_with_map_train_20:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_no_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_no_trust_with_map_test.parquet" --output_folder_path="models/no_trust_with_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

with_trust_no_map_train_20:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_with_trust_no_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_with_trust_no_map_test.parquet" --output_folder_path="models/with_trust_no_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

with_trust_with_map_train_20:
	poetry run python -m condition_prediction --train_data_path="data/orderly/datasets/orderly_with_trust_with_map_train.parquet" --test_data_path="data/orderly/datasets/orderly_with_trust_with_map_test.parquet" --output_folder_path="models/with_trust_with_map_20"  --train_fraction=0.2 --train_val_split=0.8 --overwrite=False --epochs=20 --evaluate_on_test_data=True --early_stopping_patience=5 --wandb_entity=WANDB_ENTITY

train_all: no_trust_no_map_train no_trust_with_map_train with_trust_no_map_train with_trust_with_map_train no_trust_no_map_train_20 no_trust_with_map_train_20 with_trust_no_map_train_20 with_trust_with_map_train_20

# Example workflow for running everything: no trust, with map
# extract, clean, fp
example_workflow_prep_data: paper_extract_uspto_no_trust, paper_gen_uspto_no_trust_with_map, fp_no_trust_with_map_test, fp_no_trust_with_map_train

# change env to have TF
example_workflow_train_model: no_trust_with_map_train

