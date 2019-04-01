
branch=$(shell git rev-parse --abbrev-ref HEAD)

img2=duckietown/gym-duckietown-server-python2:$(branch)
img3=duckietown/gym-duckietown-server-python3:$(branch)
img2-ros=duckietown/gym-duckietown-server-python2-ros:$(branch)

all:
	@echo ## Containerized Python 2 support
	@echo
	@echo To build all containers:
	@echo
	@echo    make build-docker-python2
	@echo    make build-docker-python2-ros
	@echo
	@echo To push to:
	@echo
	@echo  $(img2)
	@echo  $(img2-ros)
	@echo
	@echo To develop in the container (deps in container, code in this dir), use:
	@echo
	@echo    make shell-docker-python2-ros
	@echo
	@echo Inside, remember to start  launch-xvfb


build:
	$(MAKE) build-docker-python3
	$(MAKE) build-docker-python2
	$(MAKE) build-docker-python2-ros

push:
	$(MAKE) push-docker-python3
	$(MAKE) push-docker-python2
	$(MAKE) push-docker-python2-ros


build-docker-python2:
	docker build -t $(img2) -f docker/AIDO1/server-python2/Dockerfile .

push-docker-python2:
	docker push $(img2)



build-docker-python3:
	docker build -t $(img3) -f docker/AIDO1/server-python3/Dockerfile .

push-docker-python3:
	docker push $(img3)

build-docker-python2-ros:
	docker build -t $(img2-ros) -f docker/AIDO1/server-python2-ros/Dockerfile .

push-docker-python2-ros:
	docker push $(img2-ros)

shell-docker-python2-ros:
	@echo Running the image with the current directory - happy development
	@echo
	@echo Need to run `launch-xvfb` inside.
	@echo
	docker run -it -v $(PWD):/workspace/gym-duckietown --network host -w /workspace/gym-duckietown duckietown/gym-duckietown-server-python2-ros bash


other_deps:
	apt install x11-apps
