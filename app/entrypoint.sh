#!/usr/bin/env bash
set -e

NUM_WORKERS="${NUM_WORKERS:-4}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "SERVER_HOST=${SERVER_HOST}"
echo "SERVER_PORT=${SERVER_PORT}"

export ENVIRONMENT="$(echo "${ENVIRONMENT:-TEST}" | tr '[:lower:]' '[:upper:]')"
echo "ENVIRONMENT=${ENVIRONMENT}"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "APP_DIR=${APP_DIR}"
cd "$APP_DIR"

# For production environments, it's recommended to utilize Gunicorn with the Uvicorn worker class
# https://github.com/Kludex/uvicorn-worker?tab=readme-ov-file#deployment
echo "Starting API in ${ENVIRONMENT} with ${NUM_WORKERS} workers on ${SERVER_HOST}:${SERVER_PORT}"
gunicorn api.main:app -w "$NUM_WORKERS" -k uvicorn_worker.UvicornWorker --bind "${SERVER_HOST}:${SERVER_PORT}"
