FROM python:3.10-slim-buster as orderly_base

RUN apt-get update
RUN apt-get install -y libpq-dev gcc make

RUN adduser --disabled-password worker
USER worker

ENV PATH="/home/worker/.local/bin:${PATH}"

WORKDIR /home/worker/repo/
ADD pyproject.toml poetry.lock README.md /home/worker/repo/

RUN python -m pip install -U pip
RUN python -m pip install poetry
RUN python -m poetry install

ADD Makefile /home/worker/repo/
ADD orderly/ /home/worker/repo/orderly/
ADD tests/ /home/worker/repo/tests/

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

FROM orderly_base as orderly_test

CMD ["make", "pytest"]

FROM orderly_base as orderly_black

CMD ["make", "black"]

FROM ubuntu:20.04 as orderly_download
RUN apt-get update && apt-get install -y make curl unzip

FROM orderly_download as orderly_download_linux

RUN adduser --disabled-password worker
USER worker
WORKDIR /app
ADD Makefile /app

CMD ["make", "_linux_get_ord"]

FROM orderly_download as orderly_download_root

WORKDIR /app
ADD Makefile /app

CMD ["make", "_root_get_ord"]

FROM continuumio/miniconda3 as rxnmapper_base

RUN conda create -n rxnmapper python=3.6 -y
SHELL ["conda", "run", "-n", "rxnmapper", "/bin/bash", "-c"]
RUN pip install rxnmapper
RUN pip install rdkit-pypi

RUN echo 'conda activate rxnmapper' >> /root/.bashrc
ENTRYPOINT [ "/bin/bash", "-l"]
