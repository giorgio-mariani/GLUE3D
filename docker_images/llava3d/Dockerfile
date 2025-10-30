FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Basic system and Python dev setup
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.10 python3.10-dev python3.10-distutils \
    build-essential \
    git \
    curl \
    cmake \
    ninja-build \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set python and pip as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

# Set working directory
WORKDIR /ShapeLLM

# Install pytorch
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install cv2 deps
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy current directory contents into the container
COPY ./LLaVA-3D .
COPY ./requirements.txt .

# Install the current directory as a Python package
RUN pip install -r requirements.txt
RUN pip install -e .
RUN pip install datasets
RUN pip install OpenEXR
COPY ./playground ./playground
