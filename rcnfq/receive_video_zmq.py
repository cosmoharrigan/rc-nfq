'''Receive streaming video from the robot using ZeroMQ
'''

import io
import socket
import struct
from PIL import Image
from scipy.misc import imread
import numpy as np
import zmq
import time

# Configure the following parameter:
IP_ADDRESS = "192.168.0.56"

# Setup SUBSCRIBE socket
context = zmq.Context()
zmq_socket = context.socket(zmq.SUB)
zmq_socket.setsockopt(zmq.SUBSCRIBE, b'')
zmq_socket.setsockopt(zmq.CONFLATE, 1)
zmq_socket.connect("tcp://{}:5557".format(IP_ADDRESS))

i = 0
try:
    while True:
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        payload = zmq_socket.recv()       
        image_stream.write(payload)
        
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = imread(image_stream)

        # Do something with the image
        print(np.round(np.mean(image), 0))

        i += 1
finally:
    pass
