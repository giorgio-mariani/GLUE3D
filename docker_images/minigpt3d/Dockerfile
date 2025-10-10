FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Basic system and Python dev setup
RUN apt-get update
RUN apt-get install -y \
    software-properties-common \
    python3.10 python3.10-venv python3.10-dev python3.10-distutils \
    build-essential \
    git \
    curl \
    cmake \
    ninja-build && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN apt-get remove -y python3-blinker

# Install opencv dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

# Set up python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

# Fix pip version
RUN pip install --upgrade pip==25.1

# Set working directory
WORKDIR /workdir

# Install pytorch
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Copy current directory contents into the container
COPY ./MiniGPT-3D .
COPY ./requirements.txt .

RUN pip install -r requirements.txt
RUN cp modeling_phi.py /usr/local/lib/python3.10/dist-packages/transformers/models/phi/

RUN mkdir -p ./data/anno_data && mkdir -p ./data/modelnet40_data && mkdir -p ./data/objaverse_data
ENV MINIGPT3D_ROOT=/workdir

COPY download_weights.py .
RUN PYTHONPATH=. python download_weights.py
