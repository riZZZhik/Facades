# to build use: docker image build -t oneonwar:jptr .
# to run use: nvidia-docker run --gpus all -it --rm -v ~/:/USER_FILES/ -p 4958:4958 oneonwar:jptr

# Who did it?
LABEL maintainer="t.me/riZZZhik"

# User arguments (Note that TensorFlow does not support CUDA 10.1)
ARG UBUNTU_VERSION=18.04
ARG CUDA_VERSION=10.0

# Load base image
FROM nvidia/cuda:${CUDA_VERSION:-10.0}-cudnn7-runtime-ubuntu${UBUNTU_VERSION:-18.04}

# FIX: I don't know fucking why is it needed, maybe nvidia was mistaken, anyway, DO NOT DELETE THIS SHIT
#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# User arguments
# py3 only
ARG PYTHON_VERSION=3.6

# Python docker arguments
ARG PYTHON_SUFFIX=${PYTHON_VERSION:-3.6}
ARG PYTHON=python${PYTHON_VERSION}
ARG PIP=pip${PYTHON_VERSION}

# Install python
RUN apt-get update && apt-get install -y python$PYTHON_VERSION \
                                         python3-pip \
                                         python3-dev

# Add python to bash
ENV PATH="/usr/bin:$PATH"

################################################################
######################### USER SCRIPT ##########################
################################################################
# Copy and install python packages
# FIXME: Could not open requirements file: [Errno 2] No such file or directory: '/tmp/requirements.txt'
#COPY requirments.txt /tmp/
#RUN pip3 --no-cache-dir install -r /tmp/requirements.txt

RUN pip3 --no-cache-dir install jupyterlab

# Run jupyter lab
WORKDIR /
CMD jupyter lab --ip 0.0.0.0 --port 4958 --no-browser --allow-root