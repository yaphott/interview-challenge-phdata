FROM ubuntu:24.04

ARG NUM_WORKERS=4
ARG SERVER_HOST=127.0.0.1
ARG SERVER_PORT=8000
ARG ENVIRONMENT=PROD
ARG MODEL_PATH=/app/model/model.pkl
ARG DEMOGRAPHICS_PATH=/app/data/zipcode_demographics.csv

RUN apt-get update && apt-get install -y ca-certificates curl gnupg software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsLS --proto '=https' --tlsv1.2 "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" -o /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p /opt/conda \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && rm /tmp/miniconda3.sh \
    && /opt/conda/bin/conda clean -afy

COPY ./app/environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -n api-env -f /tmp/environment.yml -y \
    && rm /tmp/environment.yml

WORKDIR /app
COPY ./app/api/ ./api/
COPY ./app/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

COPY ./model/ ./model/
COPY ./data/ ./data/

ENV NUM_WORKERS=${NUM_WORKERS}
ENV SERVER_HOST=${SERVER_HOST}
ENV SERVER_PORT=${SERVER_PORT}
ENV ENVIRONMENT=${ENVIRONMENT}
ENV MODEL_PATH=${MODEL_PATH}
ENV DEMOGRAPHICS_PATH=${DEMOGRAPHICS_PATH}

CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "api-env", "bash", "/app/entrypoint.sh"]
