FROM ubuntu:20.04 as orderly_download

RUN apt-get update && apt-get install -y make curl unzip

WORKDIR /app
ADD Makefile /app

CMD ["make", "get_ord"]

FROM ubuntu:20.04 as orderly_download_safe

RUN apt-get update && apt-get install -y make curl unzip

RUN adduser worker

WORKDIR /app
ADD Makefile /app

RUN chown worker:worker /app
USER worker

CMD ["make", "get_ord_safe"]

FROM ubuntu:20.04 as orderly_download_mounted

RUN apt-get update && apt-get install -y make curl unzip

RUN adduser worker
USER worker
WORKDIR /home/worker

ENV PATH="/home/worker/.local/bin:${PATH}"

ADD pyproject.toml poetry.lock Makefile README.md /home/worker/
ADD orderly /home/worker/orderly/

RUN python -m pip install -U pip
RUN python -m pip install poetry

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

FROM python:3.10-slim-buster as orderly_base

RUN apt-get update
RUN apt-get install -y libpq-dev gcc make curl unzip

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


FROM ubuntu:20.04 as debug_orderly_download_safe

RUN apt-get update && apt-get install -y make curl unzip

RUN adduser worker
WORKDIR /app
ADD Makefile /app
RUN chown worker:worker /app
RUN chown worker:worker /app/Makefile

USER worker

CMD ["bash"]
# CMD ["make", "debug_get_ord"]