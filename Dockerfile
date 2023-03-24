FROM ubuntu:20.04 as orderly_download

WORKDIR /tmp
ADD Makefile /tmp/

CMD ["make", "get_ord"]

FROM python:3.10-slim-buster as orderly_base

RUN apt-get update
RUN apt-get install -y libpq-dev gcc

RUN adduser worker
USER worker
WORKDIR /home/worker

ENV PATH="/home/worker/.local/bin:${PATH}"

ADD pyproject.toml poetry.lock Makefile README.md /home/worker/
ADD orderly /home/worker/orderly/

RUN python -m pip install -U pip
RUN python -m pip install poetry
RUN python -m poetry install

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

FROM orderly_base as orderly_extract

CMD ["bash"]

FROM orderly_base as orderly_clean

CMD ["bash"]
