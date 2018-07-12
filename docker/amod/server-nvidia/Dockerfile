# requires nvidia-docker2

# before running this container, use this command once: xhost +si:localuser:root
# the run the container like so (to connect via loacalhost):

# docker build -t gym-duckietown-server .

# then

# docker run \
# --runtime=nvidia -ti --rm -e DISPLAY \
# -v /tmp/.X11-unix:/tmp/.X11-unix \
# --name gym-server -p 8902:8902 -p 5558:5558 \
# gym-duckietown-server


FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

# TODO: double-check that we really need all of these for just the server
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         bzip2 \
         curl \
         ca-certificates \
         xvfb \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy pyyaml scipy pyzmq && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone https://github.com/duckietown/duckietown-slimremote && cd duckietown-slimremote && pip install -e . && cd ..
RUN git clone https://github.com/duckietown/gym-duckietown && cd gym-duckietown && pip install -e .


EXPOSE 5558 8902
ENTRYPOINT ["duckietown-start-gym"]

