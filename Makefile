

build-docker-python2:
	docker build -t $(img)-f docker/amod/server-python2/Dockerfile .
