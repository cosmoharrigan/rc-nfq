'''
Receives a live video stream from the robot using ZeroMQ
'''

import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
import zmq

# Configure the following parameter:
IP_ADDRESS = "192.168.0.56"

cv2.startWindowThread()
cv2.namedWindow('Robot Camera', cv2.WINDOW_NORMAL)

# Setup SUBSCRIBE socket
context = zmq.Context()
zmq_socket = context.socket(zmq.SUB)
zmq_socket.setsockopt(zmq.SUBSCRIBE, b'')
zmq_socket.setsockopt(zmq.CONFLATE, 1)
zmq_socket.connect("tcp://{}:5557".format(IP_ADDRESS))

try:
    i = 0
    while True:
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        payload = zmq_socket.recv()       
        image_stream.write(payload)
        
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = Image.open(image_stream)
        downsampled_image = np.array(image.convert('L'))

        cv2.imshow('Robot Camera', downsampled_image)
        cv2.waitKey(1)
        i += 1
finally:
    pass
