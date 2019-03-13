FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN conda install numpy pyyaml scipy ipython mkl mkl-include && conda clean -ya

WORKDIR /gym-duckietown

COPY . .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v -e .


#FROM nvidia/cuda:9.1-runtime-ubuntu16.04
#
#RUN apt-get update -y && apt-get install -y --no-install-recommends \
#         git \
#         xvfb \
#         bzip2 \
#         python-pip \
#         python-setuptools\
#         freeglut3-dev \
#          python-subprocess32 python-matplotlib python-yaml python-opencv
##         && \
##     rm -rf /var/lib/apt/lists/*
#
#WORKDIR /workspace
#
#
#
#COPY docker/AIDO1/server-python2/requirements.txt /requirements.txt
#RUN pip install -r /requirements.txt
#
#
#RUN pip install -e git+https://github.com/duckietown/duckietown-slimremote.git#egg=duckietown-slimremote
#
#
#EXPOSE 5558 8902
#
#
#
#ADD . gym-duckietown
#
#RUN cd gym-duckietown && python setup.py develop --no-deps
#
#COPY docker/AIDO1/server-python2/launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb
#COPY docker/AIDO1/server-python2/launch-xvfb /usr/bin/launch-xvfb
#
#CMD launch-gym-server-with-xvfb
