## WORKING NON-NVIDIA IMAGE

#### To use the last Docker Hub upload:

# docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix duckietown/gym-duckietown-server

#### To build it yourself:

# docker build -t gym-duckietown-server3 .

#### then run with

# docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gym-duckietown-server3

FROM ubuntu:16.04

# TODO: double-check that we really need all of these for just the server
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         bzip2 \
         psmisc \
         ca-certificates \
         xvfb \
         mesa-utils \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN sh miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# I moved the pip requirements up here so we can cache them in this layer and don't
# need to rebuild them when the git repo has changed. Also I don't know why the conda pkg
# "pyzmq" isn't doing anything. Pip doesn't recognize it and installs the pkg 'zmq'.

RUN conda install numpy pyyaml scipy pyzmq && \
    pip install zmq pyglet opencv-python scikit-image gym && \
    conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

# TODO: This should be possible, but fails with ModuleNotFoundException at runtime.
# Need to figure out why this doesn't work at some point.
#RUN pip install https://github.com/duckietown/duckietown-slimremote/zipball/master \
#                https://github.com/duckietown/gym-duckietown/zipball/master

RUN pip install -e git+https://github.com/duckietown/duckietown-slimremote.git#egg=duckietown-slimremote 
RUN pip install -e git+https://github.com/duckietown/gym-duckietown.git#egg=gym-duckietown

COPY launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb
RUN chmod +x /usr/bin/launch-gym-server-with-xvfb

#RUN Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1

EXPOSE 5558 8902
CMD launch-gym-server-with-xvfb
