#### working. build with

# docker build -t gym-duckietown-server3 .

#### then run with

# docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gym-duckietown-server3

FROM ubuntu:16.04

# TODO: double-check that we really need all of these for just the server
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         bzip2 \
         curl \
         ca-certificates \
         xvfb \
         mesa-utils \
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
RUN git clone https://github.com/duckietown/gym-duckietown && cd gym-duckietown && pip install -e . && cd ..
COPY launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb
RUN chmod +x /usr/bin/launch-gym-server-with-xvfb

#RUN Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1


EXPOSE 5558 8902
ENTRYPOINT ["launch-gym-server-with-xvfb"]

