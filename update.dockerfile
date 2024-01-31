ARG BASE_DOCKER
# Dockerfile that updates the container with new code.
# SageMaker PyTorch image
FROM ${BASE_DOCKER}

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY . /opt/ml/code/