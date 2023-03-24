current_dir = $(shell pwd)
uid = $(shell id -u)
gid = $(shell id -g)

get_ord:
	curl -L -o /tmp/repo.zip https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip
	unzip -o /tmp/repo.zip -d /tmp_data
	rm /tmp/repo.zip

build_download_ord:
	docker image build --target orderly_download --tag ord_download .

run_download_ord:
	docker run -v $(current_dir)/data:/tmp_data -u $(uid):$(gid) ord_download

build_orderly:
	docker image build --tag orderly .

run_orderly:
	docker run -v $(current_dir)/data:/tmp_data -u $(uid):$(gid) -it orderly

get_paper:
	docker run --rm --volume $(current_dir)/paper:/data --user $(uid):$(gid) --env JOURNAL=joss openjournals/inara
	rm $(current_dir)/paper/paper.jats
