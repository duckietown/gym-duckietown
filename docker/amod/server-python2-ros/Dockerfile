FROM duckietown/gym-duckietown-server-python2
# ROS INSTALL -------------------------
# install packages

RUN apt-get update && apt-get install -q -y \
        dirmngr \
        gnupg2 \
        apt-utils \
        apt-file \
        lsb-release
#    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116


RUN apt-get update
RUN apt-get install -y \
    ros-kinetic-ros-base \
    ros-kinetic-rosbag \
    ros-kinetic-common-msgs \
    ros-kinetic-catkin


##RUN apt-get install -q -y python-catkin-pkg
#RUN apt-get install -q -y ros-kinetic-catkin
#
#RUN apt-get install -y ros-kinetic-catkin
#RUN apt-get install -y ros-kinetic-common-msgs
#RUN apt-get install -y ros-kinetic-roslib
#RUN apt-get install -y ros-kinetic-rosbag
#RUN pip install rospkg
ENV PYTHONPATH="/opt/ros/kinetic/lib/python2.7/dist-packages/:${PYTHONPATH}"
# ------------------------------------------


RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# not sure
RUN rosdep init
RUN rosdep update

ENV DISPLAY=:99
