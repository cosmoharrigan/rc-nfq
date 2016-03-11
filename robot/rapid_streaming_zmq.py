'''Streams video from the robot using ZeroMQ.

This example uses a Raspberry Pi camera and requires the following libraries to
be installed on the Raspberry Pi:
- https://picamera.readthedocs.org/en/release-1.10/
- http://zeromq.org/
'''

import io
import socket
import struct
import time
import picamera
import zmq

# Setup ZeroMQ PUBLISH socket
context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)
zmq_socket.bind("tcp://0.0.0.0:5557")

try:
    with picamera.PiCamera() as camera:
        camera.resolution = ((64, 64))
        camera.framerate = 30
        time.sleep(2)
        start = time.time()

        # Keep track of frames sent to calculate actual frame rate
        n = 0
        
        stream = io.BytesIO()
        # Capture from the video port
        for foo in camera.capture_continuous(stream, 'jpeg',
                                             use_video_port=True):
            stream.seek(0)
            
            zmq_socket.send(stream.read())
            n += 1

            stream.seek(0)
            stream.truncate()
finally:
    pass
