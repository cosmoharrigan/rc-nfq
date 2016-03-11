"""
API for running Deep Q-learning experiments, where the state input is from a
separate video stream.
"""

import json, time
from flask import Flask, Response, jsonify, render_template, request, make_response
import gevent
from gevent.wsgi import WSGIServer
from gevent.queue import Queue
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import cycle
import StringIO
from environments import VisionRobot
from experiments import ConvolutionalNFQExperiment

# Configuration. Set the following values:
IP_ADDRESS = "192.168.0.56"
EPSILON = 0.20
SAVED_WEIGHTS = None

########### Receive video ##########
import io
import socket
import struct
from PIL import Image
from scipy.misc import imread
import numpy as np
import zmq
import time

# Setup SUBSCRIBE socket
context = zmq.Context()
zmq_socket = context.socket(zmq.SUB)
zmq_socket.setsockopt(zmq.SUBSCRIBE, b'')
zmq_socket.setsockopt(zmq.CONFLATE, 1)
zmq_socket.connect("tcp://{}:5557".format(IP_ADDRESS))

def get_image():
    image_stream = io.BytesIO()
    payload = zmq_socket.recv()       
    image_stream.write(payload)
    
    # Rewind the stream, open it as an image with PIL and do some
    # processing on it
    image_stream.seek(0)
    image = Image.open(image_stream)

    # Convert to grayscale
    grayscale_image = np.array(image.convert('L'))
    
    # Normalize
    grayscale_image -= 128
    grayscale_image = grayscale_image.astype(np.float32) / 128

    return grayscale_image
####################################

NB_EPISODES = 1
MAX_STEPS_PER_EPISODE = 5001

ENVIRONMENT = VisionRobot()
EXPERIMENT = \
    ConvolutionalNFQExperiment(env=ENVIRONMENT,
                               nb_episodes=NB_EPISODES,
                               max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                               nb_samples=100000,
                               sliding_window=100000,
                               target_network_update_freq=20,
                               lr=0.0001)

# Load the saved weights from the previous experiment
if SAVED_WEIGHTS:
    EXPERIMENT.nfq.Q.load_weights(SAVED_WEIGHTS)
    EXPERIMENT.nfq.Q_target.load_weights(SAVED_WEIGHTS)

EXPERIMENT_LENGTH = NB_EPISODES * MAX_STEPS_PER_EPISODE
HISTORY = np.zeros((EXPERIMENT_LENGTH, 4))
UPDATE_HISTORY = np.zeros((EXPERIMENT_LENGTH, 3))  # to store: a, r, r_cum
STATE_HISTORY = np.zeros((EXPERIMENT_LENGTH,
                          ENVIRONMENT.state_dim[0],
                          ENVIRONMENT.state_dim[1],
                          ENVIRONMENT.state_dim[2]))
AVG_REWARD = np.zeros((EXPERIMENT_LENGTH, 1))
N = 0

CURRENT_STATE = None

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return '<meta http-equiv="refresh" content="5; URL=/"> <img src="/plot">'

@app.route('/health/', methods=['GET'])
def health():
    return '200 OK'

@app.route('/state/', methods=['GET'])
def state():
    image = Image.fromarray(CURRENT_STATE)
    output = StringIO.StringIO()
    image.save(output, format='png')
    contents = output.getvalue()
    output.close()

    response = make_response(contents)
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route('/log/', methods=['POST'])
def log():
    """Log diagnostic details about the robot
    """
    global HISTORY
    global N

    if N < EXPERIMENT_LENGTH:
        HISTORY[N, 0] = int(request.form['speed_left'])
        HISTORY[N, 1] = int(request.form['speed_right'])
        HISTORY[N, 2] = int(request.form['position_left'])
        HISTORY[N, 3] = int(request.form['position_right'])
        N += 1

    return "OK"

@app.route('/update/', methods=['POST'])
def update():
    """Record a (s, a, r, s') tuple. We only receive a partial tuple from the
    robot:
        (_, a, r, _)
    and we need to obtain s' from the video camera stream, and s from the
    history of the previous update, where s = s_prime_previous
    """
    global UPDATE_HISTORY
    global N
    global EXPERIMENT
    global CURRENT_STATE

    if N < EXPERIMENT_LENGTH:
        # Deserialize (s, a, r, s')
        a = int(request.form['a'])
        r = float(request.form['r'])

        PREVIOUS_STATE = CURRENT_STATE
        CURRENT_STATE = get_image()

        # Get s from the previous step s'
        if N > 0:
            # Get s' from the video camera stream
            s = PREVIOUS_STATE
            s_prime = CURRENT_STATE

            # Record the tuple in the history
            if N > 0:
                r_cum = UPDATE_HISTORY[N - 1, 2] + r
            else:
                r_cum = 0

            UPDATE_HISTORY[N, :] = np.array((a, r, r_cum))
            STATE_HISTORY[N, :] = s.reshape(STATE_HISTORY.shape[1],
                                            STATE_HISTORY.shape[2],
                                            STATE_HISTORY.shape[3])

            if N > 20:
                avg_reward = np.mean(UPDATE_HISTORY[N-20:N, 1])
            else:
                avg_reward = 0
            AVG_REWARD[N] = avg_reward

            EXPERIMENT.update(s, a, r, s_prime)

    N += 1

    return "OK"

@app.route('/plot/', methods=['GET'])
def plot():
    from matplotlib import pyplot as plt
    fig = plt.Figure()

    ax5 = fig.add_subplot(411)
    ax5.plot(UPDATE_HISTORY[0:N, 1])
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Reward')
    ax5.set_title('Reward over time')

    ax6 = fig.add_subplot(412)
    ax6.plot(AVG_REWARD[0:N])
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Average reward')
    ax6.set_title('Average reward over time')
    
    ax7 = fig.add_subplot(413)
    ax7.plot(EXPERIMENT.nfq._q_predicted[0:N], 'o')
    ax7.set_ylim([-2, 8])
    ax7.set_xlabel('Time step')
    ax7.set_ylabel('Predicted Q-value for chosen action')
    ax7.set_title('Predicted Q-value over time')
    
    ax8 = fig.add_subplot(414)
    ax8.plot(EXPERIMENT.nfq._loss_history[0:EXPERIMENT.nfq.k], 'o')
    ax8.set_ylim([-3, 3])
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('NFQ loss')
    ax8.set_title('NFQ loss over time')

    fig.set_size_inches(16, 16)
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route('/policy/', methods=['GET', 'POST'])
def policy():
    """Given a state returns an action, using the current policy derived from
    the Q-function
    """
    s = CURRENT_STATE
    
    if s is None or EXPERIMENT.state == 'EXPERIMENT ENDED':
        a = 0
    else:
        # Epsilon-greedy action selection
        if np.random.random() < EPSILON:
            print('Choosing epsilon-greedy random action.')
            a = np.random.choice(ENVIRONMENT.actions)
        else:
            a = EXPERIMENT.nfq.greedy_action(s)

    return str(a)

if __name__ == "__main__":
    print('Starting RC-NFQ server.')
    app.debug = True
    server = WSGIServer(("0.0.0.0", 5000), app)
    server.serve_forever()
