

img2=duckietown/gym-duckietown-server-python2


build:
	$(MAKE) build-docker-python2
	$(MAKE) build-docker-python2-ros


build-docker-python2:
	docker build -t $(img2) -f docker/amod/server-python2/Dockerfile .

push-docker-python2:
	docker push $(img2)

img2-ros=duckietown/gym-duckietown-server-python2-ros

build-docker-python2-ros:
	docker build -t $(img2-ros) -f docker/amod/server-python2-ros/Dockerfile .


shell-docker-python2-ros:
	docker run -it -v $PWD:/workspace/gym-duckietown duckietown/gym-duckietown-server-python2-ros bash


