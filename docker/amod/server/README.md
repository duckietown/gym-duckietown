# duckietown/gym-duckietown-server

[![Docker Build Status](https://img.shields.io/docker/build/duckietown/gym-duckietown.svg)](https://hub.docker.com/r/duckietown/gym-duckietown-server)

## Running the container

To use the last Docker Hub upload:

```
docker run -ti --name gym-duckietown -p 8902:8902 -p 5558:5558 \
       -e DISPLAY=$DISPLAY \
       -e DUCKIETOWN_CHALLENGE=LF \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       duckietown/gym-duckietown-server
```

## Building the image

To build it yourself cd into the root directory of this repository and run:

`docker build --file ./docker/amod/server/Dockerfile --tag gym-duckietown-server3 .`
