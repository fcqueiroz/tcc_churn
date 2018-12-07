FROM continuumio/miniconda3

RUN apt-get update --fix-missing && apt-get install -y make gcc graphviz && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Create the same environment available in *.yml file
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a -y

# Pull the environment name out of the environment.yml (not working so well)
#RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
#ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
RUN echo "source activate tcc_churn" > ~/.bashrc
ENV PATH /opt/conda/envs/tcc_churn/bin:$PATH

# Prepare Jupyter Notebook for using port 8888, not using a browser and not requiring a password
RUN jupyter notebook --generate-config && rm /root/.jupyter/jupyter_notebook_config.py
ADD jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

ADD setup.py .
RUN pip install -e .

WORKDIR /tcc_churn
VOLUME /tcc_churn

LABEL maintainer="Fernanda Coelho de Queiroz <fernanda.cdqueiroz@gmail.com>"
LABEL app_name="tcc_churn"
LABEL version="0.3"

ENV APP_NAME='tcc_churn'
ENV APP_VERSION='0.3'
ENV APP_MODELS='/tcc_churn/models/'
ENV APP_REPORTS='/tcc_churn/reports/'
ENV DATA_EXTERNAL='/tcc_churn/data/external/'
ENV DATA_INTERIM='/tcc_churn/data/interim/'
ENV DATA_PROCESSED='/tcc_churn/data/processed/'
ENV DATA_RAW='/tcc_churn/data/raw/'
