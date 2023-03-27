current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)
download_path=ord/

get_ord_safe:
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /app
	mkdir -p /app/data/ord
	cp -a /app/ord-data-main/data/. /app/data/ord
	rm /app/repo.zip

build_download_ord:
	docker image build --target orderly_download_safe --tag ord_download_safe .

run_download_ord:
	docker run -u $(uid):$(gid) -it --name tmp_download_ord_safe ord_download_safe
	docker cp tmp_download_ord_safe:/app/data .
	docker rm -f tmp_download_ord_safe

build_orderly:
	docker image build --tag orderly .

run_orderly:
	docker run -v $(current_dir)/data:/tmp_data -u $(uid):$(gid) -it orderly

get_paper:
	docker run --rm --volume $(current_dir)/paper:/data --user $(uid):$(gid) --env JOURNAL=joss openjournals/inara
	rm $(current_dir)/paper/paper.jats

prune:
	docker system prune -a --volumes

clear:
	sudo rm -rf ./data/

extract:
	poetry run python -m orderly.extract

black:
	poetry run python -m black .

get_test_data:
	poetry run python -m orderly.extract --data_path=orderly/data/ord_test_data --output_path=orderly/data/extracted_ord_test_data --overwrite=False

debug_build:
	docker image build --tag orderly_download_mounted .

debug_run:
	docker run -v $(current_dir)/data:/tmp_data -u $(uid):$(gid) -it orderly_download_mounted

linux_download_ord:
	docker image build --target orderly_download_linux --tag orderly_download_linux .
	docker run -v $(current_dir)/data:/tmp_data -u $(uid):$(gid) -it orderly_download_linux
	# docker rm -f orderly_download_linux

linux_get_ord:
	mkdir -p /tmp_data/${download_path}
	touch /tmp_data/${download_path}/tst_permissions_file.txt
	rm /tmp_data/${download_path}/tst_permissions_file.txt
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /app
	cp -a /app/ord-data-main/data/. /tmp_data/${download_path}
	rm /app/repo.zip

sudo_download_ord:
	docker image build --target orderly_download --tag ord_download .
	docker run -v $(current_dir)/tmp_data:/data ord_download
	docker rm -f ord_download
	sudo chown -R $(uid):$(gid) $(current_dir)/data

sudo_get_ord:
	mkdir -p /tmp_data/${download_path}
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /tmp_data/${download_path}
	rm /app/repo.zip