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

# Set python and pip as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

# Fix pip version
RUN pip install --upgrade pip==25.1

# Set working directory
WORKDIR /workdir

# Install pytorch
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Pointnet++
RUN pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Copy current directory contents into the container
COPY ./ShapeLLM .
RUN mkdir ./checkpoints
RUN mkdir ./checkpoints/recon
RUN curl -L https://huggingface.co/qizekun/ReConV2/resolve/main/zeroshot/large/best_lvis.pth -o ./checkpoints/recon/large.pth


# Install the current directory as a Python package
RUN pip install -e .
RUN pip install datasets==4.0.0
