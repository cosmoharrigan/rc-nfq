## RC-NFQ: Regularized Convolutional Neural Fitted Q Iteration

### A batch algorithm for deep reinforcement learning. Incorporates dropout regularization and convolutional neural networks with a separate target Q network.

[Follow @cosmosquared](https://twitter.com/cosmosquared)

This algorithm extends the following techniques:

- Riedmiller, Martin. "Neural fitted Q iteration-first experiences with a data efficient neural reinforcement learning method." Machine Learning: ECML 2005. Springer Berlin Heidelberg, 2005. 317-328.

- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

- Lin, Long-Ji. "Self-improving reactive agents based on reinforcement learning, planning and teaching." Machine learning 8.3-4 (1992): 293-321.

## Project Status

This project is still a work in progress and is not finished.

## Overview

### Creating an instance of the RC-NFQ algorithm

The [NFQ](https://github.com/cosmoharrigan/rc-nfq/rc-nfq/nfq.py) class creates an instance of the RC-NFQ algorithm for a particular agent and environment.

### Parameters

- **state_dim** - The state dimensionality. An integer if convolutional = False, a 2D tuple otherwise.
- **nb_actions** - The number of possible actions
- **terminal_states** - The integer indices of the terminal states
- **convolutional** - Boolean. When True, uses convolutional neural networks and dropout regularization. Otherwise, uses a simple MLP.
- **mlp_layers** - A list consisting of an integer number of neurons for each hidden layer. Default = [20, 20]. For convolutional = False.
- **discount_factor** - The discount factor for Q-learning.
- **separate\_target\_network** - boolean - If True, then it will use a separate Q-network for computing the targets for the Q-learning updates, and the target network will be updated with the parameters of the main Q-network every target\_network\_update_freq iterations.
- **target\_network\_update\_freq** - The frequency at which to update the target network.
- **lr** - The learning rate for the RMSprop gradient descent algorithm.
- **max_iters** - The maximum number of iterations that will be performed. Used to allocate memory for NumPy arrays. Default = 20000.
- **max\_q\_predicted** - The maximum number of Q-values that will be predicted. Used to allocate memory for NumPy arrays. Default = 100000.

### Fitting the Q network

The NFQ class has a **fit_vectorized** method, which is used to run an iteration of the RC-NFQ algorithm and update the Q function. The implementation is vectorized for improved performance.

The function requires a set of interactions with the environment. 
They consist of experience tuples of the form **(s, a, r, s_prime)**,
stored in 4 parallel arrays.

### Parameters

- **D_s** - A list of states s for each experience tuple
- **D_a** - A list of actions a for each experience tuple
- **D_r** - A list of rewards r for each experience tuple
- **D\_s\_prime** - A list of states s_prime for each experience tuple
- **num_iters** - The number of epochs to run per batch. Default = 1.
- **shuffle** - Whether to shuffle the data before training. Default = False.
- **nb\_samples** - If specified, uses nb_samples samples from the experience
             tuples selected without replacement. Otherwise, all eligible
             samples are used.
- **sliding\_window** - If specified, only the last nb_samples samples will be
                 eligible for use. Otherwise, all samples are eligible.
- **full\_batch\_sgd** - Boolean. Determines whether RMSprop will use 
                 full-batch or mini-batch updating. Default = False.
- **validation** - Boolean. If True, a validation set will be used consisting
             of the last 10% of the experience tuples, and the validation 
             loss will be monitored. Default = True.

### Setting up an experiment

An experiment consists of an [Experiment](https://github.com/cosmoharrigan/rc-nfq/api/experiment.py) definition and an [Environment](https://github.com/cosmoharrigan/rc-nfq/api/environment.py) definition. These need to be configured in the [api\_vision.py](https://github.com/cosmoharrigan/rc-nfq/api/api_vision.py) webserver.

The webserver exposes a REST resource used for communicating with the robot. An implementation of a client for a customized LEGO Mindstorms EV3 robot is provided in [client\_vision.py](https://github.com/cosmoharrigan/rc-nfq/robot/client_vision.py).

Streaming video is sent by the robot. An implementation for a customized LEGO Mindstorms EV3 robot is provided in [rapid\_streaming\_zmq.py](https://github.com/cosmoharrigan/rc-nfq/robot/rapid_streaming_zmq.py). The streaming video is then received by the server using [receive\_video\_zmq.py](https://github.com/cosmoharrigan/rc-nfq/api/receive_video_zmq.py). The video stream can be monitored using [show\_video\_zmq.py](https://github.com/cosmoharrigan/rc-nfq/api/show_video_zmq.py).

## Citation

```
@misc{rcnfq,
  author = {Harrigan, Cosmo},
  title = {RC-NFQ: Regularized Convolutional Neural Fitted Q Iteration},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cosmoharrigan/rc-nfq}}
}
```
