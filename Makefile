current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)
download_path=ord/


test_pip:
	pip install --index-url https://test.pypi.org/simple/ orderly

black:
	poetry run python -m black .

pytest:
	poetry run python -m pytest -v

get_test_data:
	poetry run python -m orderly.extract --data_path=orderly/data/ord_test_data --output_path=orderly/data/extracted_ord_test_data --overwrite=False

build_orderly:
	docker image build --target orderly_base --tag orderly_base .
	docker image build --target orderly_base_sudo --tag orderly_base_sudo .

build_orderly_rxn_test:
	docker image build --target rxnmapper_test --tag rxnmapper_test .

run_orderly_rxn_test:
	docker run -v $(current_dir)/data:/home/worker/repo/data/ -u $(uid):$(gid) -it rxnmapper_test

build_orderly_extras:
	docker image build --target orderly_test --tag orderly_test .
	docker image build --target orderly_black --tag orderly_black .

run_orderly:
	docker run -v $(current_dir)/data:/home/worker/repo/data/ -u $(uid):$(gid) -it orderly_base

run_orderly_black:
	docker run orderly_black

run_orderly_pytest:
	docker run orderly_test

run_orderly_sudo:
	docker run -v $(current_dir)/data:/home/worker/repo/data/ -it orderly_base_sudo

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

get_paper:
	docker run --rm --volume $(current_dir)/paper:/data --user $(uid):$(gid) --env JOURNAL=joss openjournals/inara
	rm $(current_dir)/paper/paper.jats

prune_docker:
	docker system prune -a --volumes

build_rxnmapper:
	docker image build --target rxnmapper_base --tag rxnmapper_base .

run_rxnmapper:
	docker run -v $(current_dir)/data:/tmp_data -it rxnmapper_base

run_python_310:
	docker run -it python:3.10-slim-buster /bin/bash

