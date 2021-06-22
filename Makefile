all:

build:
	dts build_utils aido-container-build --use-branch daffy --push



other_deps:
	apt install x11-apps

bump-upload:
	$(MAKE) bump
	$(MAKE) upload

bump: # v2
	bumpversion patch
	git push --tags
	git push

upload: # v3
	dts build_utils check-not-dirty
	dts build_utils check-tagged
	dts build_utils check-need-upload --package duckietown-gym-daffy make upload-do

upload-do:
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*

black:
	black -l 110 src
