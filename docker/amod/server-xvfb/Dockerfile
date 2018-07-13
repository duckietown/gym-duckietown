## WORKING NON-NVIDIA IMAGE

#### To use the last dockerhub upload:

# docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix duckietown/gym-duckietown-server


#### To build it yourselves:

# docker build -t gym-duckietown-server3 .

#### then run with

# docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gym-duckietown-server3

FROM ubuntu:16.04

# TODO: double-check that we really need all of these for just the server
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         bzip2 \
         curl \
         psmisc \
         ca-certificates \
         xvfb \
         mesa-utils \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

# I moved the pip requirements up here so we can cache them in this layer and don't
# need to rebuild them when the git repo has changed. Also I don't know why the conda pkg
# "pyzmq" isn't doing anything. Pip doesn't recognize it and installs the pkg 'zmq'.

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy pyyaml scipy pyzmq && \
     /opt/conda/bin/pip install zmq pyglet opencv-python scikit-image gym && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace

# this next line is manually incremented every time
# we push major changes to the gym-duckietown repo,
# so that docker forces a rebuild starting at this line
ENV force-docker-rebuild 7

RUN git clone https://github.com/duckietown/duckietown-slimremote && cd duckietown-slimremote && pip install -e . && cd ..
RUN git clone https://github.com/duckietown/gym-duckietown.git && cd gym-duckietown && git checkout amod-dev && pip install -e . && cd ..
COPY launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb
RUN chmod +x /usr/bin/launch-gym-server-with-xvfb

#RUN Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1


EXPOSE 5558 8902
ENTRYPOINT ["launch-gym-server-with-xvfb"]

