# duckietown/gym-duckietown-server

[![Docker Build Status](https://img.shields.io/docker/build/duckietown/gym-duckietown-server.svg)](https://hub.docker.com/r/duckietown/gym-duckietown-server)

## Docker images

There are two versions of the gym-duckietown-server Docker image. One for GPU, which has accelerated libraries (the default image, `duckietown/gym-duckietown-server`), and one for CPU only (`duckietown/gym-duckietown-server:slim'), which is slimmer but may run more slowly. There are also pinned versions for specific releases on [Docker Hub](https://hub.docker.com/r/duckietown/gym-duckietown-server).

## Running the container

To use the latest Docker Hub upload:

```
docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 \
       -e DISPLAY=$DISPLAY \
       -e DUCKIETOWN_CHALLENGE=LF \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       duckietown/gym-duckietown-server
```

## Building the image

To build it yourself cd into the root directory of this repository and run:

`docker build --file ./docker/amod/server/Dockerfile --tag gym-duckietown-server .`
