ARG AIDO_REGISTRY

#FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy-amd64
FROM nvidia/opengl:1.2-glvnd-devel

RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    wget curl vim git \
    && \
    rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /gym-duckietown

COPY . .

#RUN pip install -v -e .
RUN python3 -m pip install -U "pip>=21"
#RUN python3 -c "from gym_duckietown import *"



## first install the ones that do not change

COPY requirements.pin.txt .
RUN python3 -m pip install  -r requirements.pin.txt
#
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN cat .requirements.txt

RUN python3 -m pip install  -r .requirements.txt

COPY . .

RUN python3 -m pip install -v  --no-deps .

RUN python3 -m pip install pyglet==1.5.15

#   pip install -v -e .

RUN pipdeptree

ENTRYPOINT [ "xvfb-run" ]
