#FROM pytorch/pytorch
FROM nvidia/opengl:1.2-glvnd-devel
RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    python3-pip \
    python3-numpy
    python3-scipy \
    wget curl vim git \
    && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /gym-duckietown

COPY . .

#RUN pip install -v -e .

RUN python3 -m pip install pyglet==1.5.15
RUN python3 -c "from gym_duckietown import *"
