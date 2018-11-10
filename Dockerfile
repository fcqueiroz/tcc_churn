# reference: https://hub.docker.com/_/ubuntu/
FROM ubuntu:16.04

RUN apt-get update --fix-missing && apt-get install -y \
    python3 \
    python3-pip \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip3 install sklearn datetime pandas seaborn pyyaml sqlalchemy psycopg2-binary

LABEL maintainer="Fernanda Coelho de Queiroz <fernanda.cdqueiroz@gmail.com>"
LABEL app_name="tcc_churn"
LABEL version="0.1"

WORKDIR /tcc_churn

VOLUME /tcc_churn/data
VOLUME /tcc_churn/models
VOLUME /tcc_churn/reports

COPY notebooks /tcc_churn/notebooks
COPY src /tcc_churn/src
COPY Makefile /tcc_churn/
COPY config.yml /tcc_churn/

CMD make test_model