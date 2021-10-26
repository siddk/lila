"""
record.py

Records Kinesthetic Demonstrations (physically manipulating arm to demonstrate behavior).

Fully self-contained, ROS-less implementation for recording demonstrations solely using Keyboard/Joystick input.
Runs in Python3 (`conda activate lila` on NUC) w/ LibFranka C++ Controller.

OUTPUT FORMAT:
    - demonstrations :: List:
            Array[21-dim -- robot joint positions (7) + velocities (7) + torques (7)]
                (Note that you will probably not be using torques, so you just need [:14] of this array)

Automatically saved to <args.demonstration_path>/<names.pkl>
"""
import os
import pickle
import socket
import time
from subprocess import Popen, call

import numpy as np
from tap import Tap


# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


# CONSTANTS
STEP_TIME = 0.1  # Record Joint State every 0.1 seconds


# Argument Parser
class ArgumentParser(Tap):
    # fmt: off
    # Save Parameters
    name: str                                               # Name of the Demonstrations to be Collected
    demonstration_path: str = "data/lila-demonstrations"    # Path to Demonstration Storage Directory

    # Port & LibFranka Parameters
    robot_ip: str = "172.16.0.2"                            # IP address of the Panda Arm we're playing with!
    port: int = 8080                                        # Default Port to Open Socket between Panda & NUC

    # Input Device
    in_device: str = "joystick"                             # Input Device -- < joystick > (no others supported!)
    # fmt: on


def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("localhost", PORT))
    s.listen()
    conn, addr = s.accept()
    return conn


def listen2robot(conn):
    state_length = 7 + 7 + 7
    message = str(conn.recv(1024))[2:-2]
    state_str = list(message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx + 1 : idx + 1 + state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
        assert len(state_vector) == state_length
    except (AssertionError, ValueError):
        return None

    # State Vector is Concatenated Joint Positions [0 - 7], Joint Velocities [8 - 14], and Joint Torques [15 - 21]
    state_vector = np.asarray(state_vector)[0:21]
    return state_vector


def read_state(conn):
    while True:
        state_vec = listen2robot(conn)
        if state_vec is not None:
            break
    return state_vec


def record():
    # Parse Arguments
    args = ArgumentParser().parse_args()

    # Establish Initial Connection to Robot --> First connect, then spawn LibFranka Low-Level Process (./read_State)
    print("[*] Initializing Connection to Robot and Starting Read-State Loop...")
    lf_process = Popen("~/libfranka/build/examples/read_State %s %d" % (args.robot_ip, args.port), shell=True)
    conn = connect2robot(args.port)

    # Assert controller is Joystick!
    assert args.in_device == "joystick", "Only Joystick Recording is Supported for LILA Recording..."

    # Drop into While Loop w/ Controlled Recording Behavior --> Joystick Controller
    # Simple Joystick (Logitech Controller) Wrapper --> [A] starts recording, [B] stops recording,
    #                                                   [START] resets robots, [BACK] ends session
    class Joystick:
        def __init__(self):
            pygame.init()
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()

        def input(self):
            pygame.event.get()
            A, B = self.gamepad.get_button(0), self.gamepad.get_button(1)
            X = self.gamepad.get_button(2)
            START, BACK = self.gamepad.get_button(7), self.gamepad.get_button(6)

            return A, B, X, START, BACK

    print("[*] Dropping into Demonstration Loop w/ Input Device Controller = Joystick")
    print("\n[*] (A) to start recording, (B) to stop, (START) to reset, and (BACK) to end...")
    joystick, demonstrations = Joystick(), []
    while True:
        start, _, _, reset, end = joystick.input()

        # Stop Collecting Demonstrations
        if end:
            print("[*] Exiting Interactive Loop...")
            break

        # Collect a Demonstration <----> ENSURE ROBOT IS BACK-DRIVEABLE (Hit E-Stop)
        elif start:
            # Initialize Trackers
            demo = []

            # Drop into Collection Loop, recording States + Frames at STEP_TIME Intervals
            print("[*] Starting Demonstration %d Recording..." % (len(demonstrations) + 1))
            start_time = time.time()
            while True:
                # Read State
                state = read_state(conn)

                # Get Joystick Input
                _, grip, stop, _, _ = joystick.input()

                # Get Current Time and Only Record if Difference between Start and Current is > Step
                curr_time = time.time()

                # Stopping Condition
                if stop:
                    print("[*] Stopped Recording!")
                    break

                # "Firing" the Gripper
                elif grip:
                    if curr_time - start_time >= STEP_TIME:
                        print("Appending Gripper")
                        demo.append("<GRIP>")

                        # Reset Start Time
                        start_time = time.time()

                # Otherwise, just record Robot State
                else:
                    if curr_time - start_time >= STEP_TIME:
                        demo.append(state)

                        # Reset Start Time!
                        start_time = time.time()

            # Record Demo in Demonstrations
            demonstrations.append(demo)

            # Log
            print("[*] Demonstration %d\tTotal Steps: %d" % (len(demonstrations), len(demo)))

        elif reset:
            # Kill Existing Socket
            conn.close()

            # Kill LibFranka Read State
            lf_process.kill()

            # Call Reset to Home Position --> libfranka/build/examples/go_JointPosition robot_ip
            call("~/libfranka/build/examples/go_JointPosition %s" % args.robot_ip, shell=True)

            # Re-initialize LF ReadState & Socket Connection
            lf_process = Popen("~/libfranka/build/examples/read_State %s %d" % (args.robot_ip, args.port), shell=True)

            conn = connect2robot(args.port)

    # Serialize and Save Demonstrations + Images
    if not os.path.exists(args.demonstration_path):
        os.makedirs(args.demonstration_path)

    with open(os.path.join(args.demonstration_path, "%s.pkl" % args.name), "wb") as f:
        pickle.dump(demonstrations, f)

    # Cleanup
    print("[*] Shutting Down...")
    lf_process.kill()
    conn.close()


if __name__ == "__main__":
    record()
