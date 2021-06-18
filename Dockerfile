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
RUN pip3 install -U "pip>=20.2"
#RUN python3 -c "from gym_duckietown import *"



## first install the ones that do not change

COPY requirements.pin.txt .
RUN pip3 install  -r requirements.pin.txt
#
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN cat .requirements.txt

RUN pip3 install  -r .requirements.txt

COPY . .

RUN pip3 install -v  --no-deps .

RUN pip3 install pyglet==1.5.15

#   pip install -v -e .

RUN pipdeptree

ENTRYPOINT [ "xvfb-run" ]
