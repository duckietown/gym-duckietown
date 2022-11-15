# parameters
ARG REPO_NAME="gym-duckietown"
ARG DESCRIPTION="Gym Duckietown"
ARG MAINTAINER="Andrea Censi (andrea@duckietown.com)"
# pick an icon from: https://fontawesome.com/v4.7.0/icons/
ARG ICON="box"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH
ARG DISTRO=daffy
ARG DOCKER_REGISTRY=docker.io
ARG BASE_IMAGE=dt-commons
ARG BASE_TAG=${DISTRO}-${ARCH}
ARG LAUNCHER=default

# define base image
FROM ${DOCKER_REGISTRY}/duckietown/${BASE_IMAGE}:${BASE_TAG} as base

# recall all arguments
ARG ARCH
ARG DISTRO
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG ICON
ARG BASE_TAG
ARG BASE_IMAGE
ARG LAUNCHER
# - buildkit
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# <== Do not change the code above this line
# <==================================================

# install dependencies
RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    wget \
    curl \
    vim \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /gym-duckietown

## first install the ones that do not change

COPY requirements.pin.txt .
RUN python3 -m pip install  -r requirements.pin.txt

COPY requirements.* ./
RUN python3 -m pip install -r requirements.txt && \
    python3 -m pip install pyglet==1.5.15

COPY . .
RUN python3 setup.py sdist && \
    tar tfz dist/* && \
    python3 -m pip install dist/*

RUN python3 -c "import gym_duckietown;print(gym_duckietown.__file__)" && \
    python3 -c "from gym_duckietown.randomization import Randomizer; r = Randomizer()" && \
    python3 -c "import cv2" && \
    pipdeptree

ENTRYPOINT [ "xvfb-run" ]
