# Housing Project Challenge

## API Overview

This project provides an API for predicting home real estate market values based on various features. The API is built using FastAPI and serves a pre-trained machine learning model.

### Setup Instructions

1. Clone the repository and navigate to the project root directory:

    ```bash
    git clone https://github.com/yaphott/interview-challenge-phdata.git
    cd interview-challenge-phdata/api
    ```

2. Create and activate a Python virtual environment using Conda:

    ```bash
    conda env create -n housing-challenge-api -f environment.yml
    conda activate housing-challenge-api
    ```

### Usage Instructions

#### Non-Test Environments

Example starting the server with 4 workers using Uvicorn:

```bash
gunicorn 'api.main:app' -w 4 -k 'uvicorn_worker.UvicornWorker' --bind '127.0.0.1:8000'
```

#### Test Environment

For testing purposes, you can run the server with a single worker:

```bash
uvicorn 'api.main:app' --host '127.0.0.1' --port 8000 --reload
```

> **Note:** The `--reload` flag is useful during development as it automatically reloads the server on code changes.

#### Running in Docker

To run the API in a Docker container, use the provided `docker-compose.yml` file:

```bash
cd app
docker compose up --build --remove-orphans
```

This command builds the Docker image, copying the model artifacts from the [../model/](../model/) directory into the container, and starts the server, exposing the API on port 8000.
