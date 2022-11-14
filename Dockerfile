ARG DOCKER_REGISTRY

# FROM nvidia/opengl:1.2-glvnd-devel
# FROM utensils/opengl:stable
FROM ubuntu:22.04

RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    wget curl vim git \
    && \
    rm -rf /var/lib/apt/lists/*

# ARG PIP_INDEX_URL="https://pypi.org/simple"
# ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /gym-duckietown


RUN python3 -m pip install -U pip

## first install the ones that do not change

COPY requirements.pin.txt .
RUN python3 -m pip install  -r requirements.pin.txt
#
COPY requirements.* ./
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install pyglet==1.5.15

COPY . .
RUN ls
RUN python3 setup.py sdist
RUN tar tfz dist/*
RUN python3 -m pip install dist/*
RUN find /usr/local/lib/python3.8/dist-packages/gym_duckietown

RUN python3 -c "import gym_duckietown;print(gym_duckietown.__file__)"
RUN python3 -c "from gym_duckietown.randomization import Randomizer; r = Randomizer()"



RUN pipdeptree

ENTRYPOINT [ "xvfb-run" ]
