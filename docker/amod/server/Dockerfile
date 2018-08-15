FROM duckietown/gym-duckietown

RUN pip install -e git+https://github.com/duckietown/duckietown-slimremote.git#egg=duckietown-slimremote

COPY docker/amod/server/launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb

EXPOSE 5558 8902
CMD launch-gym-server-with-xvfb
