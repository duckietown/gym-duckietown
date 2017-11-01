#!/usr/bin/env python2

import zmq
import numpy
import time

SERVER_PORT=7777

# Camera image size
CAMERA_WIDTH = 100
CAMERA_HEIGHT = 100

# Camera image shape
IMG_SHAPE = (CAMERA_WIDTH, CAMERA_HEIGHT, 3)

def sendArray(socket, array):
    """Send a numpy array with metadata over zmq"""
    md = dict(
        dtype = str(array.dtype),
        shape = array.shape,
    )
    # SNDMORE flag specifies this is a multi-part message
    socket.send_json(md, flags=zmq.SNDMORE)
    return socket.send(array, flags=0, copy=True, track=False)

print('Starting up')
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % SERVER_PORT)

while True:
    print('Waiting for a command')

    msg = socket.recv_json()
    print msg

    if msg['command'] == 'reset':
        print('resetting the simulation')

    elif msg['command'] == 'action':
        print('received motor velocities')
        print(msg['values'])

    else:
        assert False, "unknown command"

    # TODO: fill in this data
    # Send world position data, etc
    # Note: the Gym client needs this to craft a reward function
    socket.send_json(
        {
            # XYZ position
            "position": [0, 0, 0],

            # Are we properly sitting inside our lane?
            "inside_lane": True,

            # Are we colliding with a building or other car?
            "colliding": False,
        },
        flags=zmq.SNDMORE
    )

    # Send a camera frame
    img = numpy.ndarray(shape=IMG_SHAPE, dtype='uint8')

    # Note: image is encoded in RGB format
    # Coordinates (0,0) are at the top-left corner
    for j in range(0, CAMERA_HEIGHT):
        for i in range(0, CAMERA_WIDTH):
            img[j, i, 0] = j # R
            img[j, i, 1] = i # G
            img[j, i, 2] = 0 # B

    sendArray(socket, img)

    time.sleep(0.05)
