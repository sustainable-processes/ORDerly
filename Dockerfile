FROM python:3.10-slim-buster as orderly_base

RUN apt-get update
RUN apt-get install -y libpq-dev gcc make

RUN adduser --disabled-password worker
USER worker

ENV PATH="/home/worker/.local/bin:${PATH}"

WORKDIR /home/worker/repo/
ADD pyproject.toml poetry.lock README.md Makefile /home/worker/repo/
ADD orderly/ /home/worker/repo/orderly/
ADD tests/ /home/worker/repo/tests/

RUN python -m pip install -U pip
RUN python -m pip install poetry
RUN python -m poetry install

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

FROM ubuntu:20.04 as orderly_download
RUN apt-get update && apt-get install -y make curl unzip

FROM orderly_download as orderly_download_linux

RUN adduser --disabled-password worker
USER worker
WORKDIR /app
ADD Makefile /app

CMD ["make", "_linux_get_ord"]

FROM ubuntu:20.04 as orderly_download_root

WORKDIR /app
ADD Makefile /app

CMD ["make", "_root_get_ord"]
