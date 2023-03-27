current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)

get_ord_safe:
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /app
	mkdir -p /app/data/ord
	cp -a /app/ord-data-main/data/. /app/data/ord
	rm /app/repo.zip

get_ord:
	curl -L -o /app/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /app/repo.zip -d /data
	rm /app/repo.zip

build_download_ord:
	docker image build --target orderly_download_safe --tag ord_download_safe .

run_download_ord:
	docker run -u $(uid):$(gid) -it --name tmp_download_ord_safe ord_download_safe
	docker cp tmp_download_ord_safe:/app/data .
	docker rm -f tmp_download_ord_safe

sudo_build_download_ord:
	docker image build --target orderly_download --tag ord_download .

sudo_run_download_ord:
	docker run -v $(current_dir)/data:/data ord_download
	sudo chown -R $(uid):$(uid) $(current_dir)

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