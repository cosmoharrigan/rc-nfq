'''Implementation of a client for a custom LEGO Mindstorms EV3 robot.

This script runs on the EV3 and requires the following custom firmware:
- http://www.ev3dev.org/
- https://github.com/topikachu/python-ev3
'''

from time import sleep
import curses
import random
from ev3dev.auto import *
import urllib
import urllib2
import random
import math

# Configure the following parameter:
IP_ADDRESS = '192.168.0.38'

front_lmotor, front_rmotor = [LargeMotor(port) for port in (OUTPUT_D, OUTPUT_A)]
rear_lmotor, rear_rmotor = [LargeMotor(port) for port in (OUTPUT_C, OUTPUT_B)]

# Check that the motors are actually connected
assert front_lmotor.connected
assert front_rmotor.connected
assert rear_lmotor.connected
assert rear_rmotor.connected

# Connect touch sensor and remote control
ts = TouchSensor(); assert ts.connected
cs = ColorSensor(); assert cs.connected

# Put the color sensor into color mode.
cs.mode = 'COL-COLOR'
colors = {0: 'none', 1: 'black', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'red',
          6: 'white', 7: 'brown'}

# Define the mapping from action indices to motor torques
MAX_SPEED = 80
SMALL_SPEED = 30
# Tuple ordering: front left, front right, rear left, rear right
actions = {0: (MAX_SPEED, MAX_SPEED, -MAX_SPEED, -MAX_SPEED),  # Forward
           1: (-MAX_SPEED, -MAX_SPEED, SMALL_SPEED, SMALL_SPEED), # Reverse
           2: (-MAX_SPEED, MAX_SPEED, MAX_SPEED, -MAX_SPEED),  # Turn Left
           3: (MAX_SPEED, -MAX_SPEED, -MAX_SPEED, MAX_SPEED),  # Turn Right
           4: (0, 0)}  # Stop

stdscr = curses.initscr()

# Robot State
state = {}
state['dc_left'] = 0
state['dc_right'] = 0

environment = {}
environment['collision'] = 0
environment['color_sensor'] = 0
environment['cumulative_reward'] = 0

front_lmotor.duty_cycle_sp = 0
front_rmotor.duty_cycle_sp = 0
front_lmotor.position = 0
front_rmotor.position = 0
front_lmotor.run_direct()
front_rmotor.run_direct()

rear_lmotor.duty_cycle_sp = 0
rear_rmotor.duty_cycle_sp = 0
rear_lmotor.position = 0
rear_rmotor.position = 0
rear_lmotor.run_direct()
rear_rmotor.run_direct()


step = 0
time = 0
action = 0  # Initial action before the first (s, a, r, s') tuple is generated
collision_this_timestep = False
color_sensor_fsm_state = 'waiting'

debug_colors = []

# Main loop
print('Starting episode.')
while True:
    # ---------------- Update instantaneous state ----------------

    # A collision can occur at any time during a time step
    ts_value = ts.value()
    if ts_value:
        collision_this_timestep = True
    environment['color_sensor'] = cs.value()

    # A finite state machine monitors track progress
    if color_sensor_fsm_state == 'waiting':
        if colors[environment['color_sensor']] == 'yellow':
            color_sensor_fsm_state = 'saw_yellow'
        elif colors[environment['color_sensor']] == 'red':
            color_sensor_fsm_state = 'saw_red'
            
    elif color_sensor_fsm_state == 'saw_yellow':
        if colors[environment['color_sensor']] == 'red':
            color_sensor_fsm_state = 'positive_reward'
        elif colors[environment['color_sensor']] != 'yellow':
            color_sensor_fsm_state = 'waiting'
            
    elif color_sensor_fsm_state == 'saw_red':
        if colors[environment['color_sensor']] == 'yellow':
            color_sensor_fsm_state = 'negative_reward'
        elif colors[environment['color_sensor']] != 'red':
            color_sensor_fsm_state = 'waiting'

    speed_left = front_lmotor.speed
    speed_right = front_rmotor.speed
    state['position_left'] = front_lmotor.position
    state['position_right'] = front_rmotor.position

    stdscr.addstr(1, 0, "Robot State", curses.A_REVERSE)
    stdscr.addstr(4, 0, "DC Left: {}".format(str(state['dc_left']).zfill(4)), curses.A_NORMAL)
    stdscr.addstr(5, 0, "DC Right: {}".format(str(state['dc_right']).zfill(4)), curses.A_NORMAL)

    stdscr.addstr(8, 0, "Cumulative Reward: {}".format(str(environment['cumulative_reward']).zfill(8), curses.A_BOLD))

    stdscr.addstr(10, 0, "Environment Information", curses.A_REVERSE)
    stdscr.addstr(12, 0, "Touch sensor: {}".format(ts_value), curses.A_NORMAL)
    stdscr.addstr(13, 0, "Color sensor: {}".format(colors[environment['color_sensor']].ljust(20)), curses.A_NORMAL)
    
    nc = min(len(debug_colors), 5)
    stdscr.addstr(14, 0, "Color history: {}".format(debug_colors[-nc:]), curses.A_NORMAL)
    
    stdscr.addstr(16, 0, "Speed Left: {}".format(str(speed_left).zfill(5)), curses.A_NORMAL)
    stdscr.addstr(17, 0, "Speed Right: {}".format(str(speed_right).zfill(5)), curses.A_NORMAL)
    stdscr.addstr(18, 0, "Position Left: {}".format(str(state['position_left']).zfill(7)), curses.A_NORMAL)
    stdscr.addstr(19, 0, "Position Right: {}".format(str(state['position_right']).zfill(7)), curses.A_NORMAL)
    stdscr.addstr(25, 0, "Step: {}".format(str(step).zfill(8)), curses.A_NORMAL)
    stdscr.addstr(26, 0, "Time: {}".format(str(time).zfill(8)), curses.A_NORMAL)
    if action is not None:
        stdscr.addstr(27, 0, "Action: {}".format(action), curses.A_NORMAL)
    stdscr.refresh()

    # Number of counts to wait between time units
    num_counts_per_time_unit = 5
    if step == num_counts_per_time_unit:
        # --------------- Perform update ---------------
        # If not first step:
        if time > 0:
            a = action

            progress_reward = 0
            if color_sensor_fsm_state == 'positive_reward':
                progress_reward = 5
                color_sensor_fsm_state = 'waiting'
            elif color_sensor_fsm_state == 'negative_reward':
                progress_reward = -2
                color_sensor_fsm_state = 'waiting'
            # Calculate reward
            r = progress_reward - 2 * collision_this_timestep #- 0.1

            # Send (_, a, r, _) tuple to server
            url = 'http://{}:5000/update/'.format(IP_ADDRESS)
            data = {'a': action,
                    'r': r}
            u = urllib2.urlopen(url, data=urllib.urlencode(data))

            # Consult the policy to obtain the next action
            url = 'http://192.168.0.38:5000/policy/'
            action_response = urllib2.urlopen(url)
            action = int(action_response.read())

            # Apply the action to the robot's actuators
            action_front_l_motor, action_front_r_motor, \
                action_rear_l_motor, action_rear_r_motor = actions[action]
            front_lmotor.duty_cycle_sp = action_front_l_motor
            front_rmotor.duty_cycle_sp = action_front_r_motor
            rear_lmotor.duty_cycle_sp = action_rear_l_motor
            rear_rmotor.duty_cycle_sp = action_rear_r_motor

        # --------------- End perform update ---------------

        # Reset step variables
        collision_this_timestep = False
        time += 1
        step = 0
        previous_state = state.copy()
    else:
        step += 1
