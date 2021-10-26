"""
imitate.py

Code for using a trained Imitation Learning model for performing language-conditioned tasks with the Panda Arm.

Run a `dry-run` AFTER training a model via: python imitate.py --name b --task_idx -1 --trial_idx -1
"""
import os
import pickle
import socket
import time
from pathlib import Path
from subprocess import DEVNULL, Popen, call

import numpy as np
import torch
from tap import Tap
from transformers import AutoModel, AutoTokenizer

from src.models import Imitation


# Suppress Tokenizers Parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


FINAL_CHECKPOINT = {"il": Path("runs/imitation-unnatural-mturk")}


class ArgumentParser(Tap):
    # fmt: off

    # User Study Parameters
    name: str = "anonymous"                         # User Name (for study)
    task_idx: int = -1                              # Task Index (which Task they're trying of the two assigned)
    trial_idx: int = -1                             # Trial Index (which Trial they're trying of the two possible)

    # Mode
    mode: str = "il"                                # Mode (Demonstrations trained on) in < il >

    # Model Parameters
    state_dim: int = 7                              # Dimensionality of Robot State (7-DoF Joint Angles)
    action_dim: int = 7                             # Dimensionality of Robot Actions (7-DoF Joint Velocities)
    hidden_dim: int = 30                            # Size of AutoEncoder Hidden Layer

    # Other Hyperparameters for Deserialization
    lr: float = 0.01                                # Learning Rate for Training
    lr_step_size: int = 200                         # Epochs to Run before LR Decay
    lr_gamma: float = 0.1                           # Learning Rate Gamma (Decay Rate)

    # Control Parameters
    action_scale: float = 1.0                       # Scalar for increasing ||joint velocities|| (actions)

    # fmt: on


# Utility Function -- Mean Pooling for Obtaining BERT "Sentence" Embeddings
def pool(output, attention_mask):
    embeddings = output[0]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embedding_sum = torch.sum(embeddings * mask, dim=1)
    mask_sum = torch.clamp(mask.sum(1), min=1e-9)
    return (embedding_sum / mask_sum).squeeze()


# Model Wrapper
class Model(object):
    def __init__(self, args):
        self.args = args

        # Initialize Model from Arguments
        print("Loading Imitation Learning Model...")
        self.model = Imitation(
            args.state_dim,
            args.action_dim,
            768,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            retrieval=True,
            run_dir=FINAL_CHECKPOINT[args.mode],
        )

        # Get Checkpoint and Restore State Dict
        checkpoints = [
            os.path.join(FINAL_CHECKPOINT[args.mode], x) for x in os.listdir(FINAL_CHECKPOINT[args.mode]) if ".ckpt" in x
        ]
        assert len(checkpoints) == 1, "Why do you have more than one valid checkpoint?"
        model_dict = torch.load(checkpoints[0], map_location="cpu")
        self.model.load_state_dict(model_dict["state_dict"])
        self.model.eval()

        # Sentence Paraphrase Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )
        self.lm = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )

        # Set Embedding to None, Initially
        self.embedding, self.exemplar = None, None

    def embed(self, instruction):
        # Tokenize & Create Embedding
        print(f"\nEncoding input instruction =>> `{instruction.lower()}`")
        encoding = self.tokenizer(instruction, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            output = self.lm(**encoding)

        # Mean Pooling
        self.embedding = pool(output, encoding["attention_mask"])

        # Unnatural Language Processing
        (idx,) = self.model.index.get_nns_by_vector(self.embedding, n=1)
        self.embedding = torch.tensor(self.model.index.get_item_vector(idx))
        self.exemplar = self.model.idx2lang[idx]

        print(f"Retrieved =>> `{self.exemplar}`")
        print("\t=>> Encoded Instruction!")

    def act(self, s):
        """ Run current state and language through policy to generate action. """
        s = torch.FloatTensor([s])
        return list(self.model(s, self.embedding.unsqueeze(0)).detach().numpy())[0]


# Controller Class
class JoystickControl(object):
    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self):
        pygame.event.get()
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y, stop = self.gamepad.get_button(2), self.gamepad.get_button(3), self.gamepad.get_button(7)
        return a, b, x, y, stop


# Robot Control Functions


# Open a Socket Connection to the Low-Level Robot Controller (Robot OR Gripper)
def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("localhost", PORT))
    s.listen()
    conn, addr = s.accept()
    return conn


# Send a Joint Velocity Command to the Low Level Controller
def send2robot(conn, qdot, gripper=False, limit=1.5):
    # Fire Gripper!
    if gripper:
        send_msg = "g,LILA,"
        conn.send(send_msg.encode())

    # Otherwise, Velocity Control
    else:
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot = np.asarray([qdot[i] * limit / scale for i in range(7)])
        send_msg = np.array2string(qdot, precision=5, separator=",", suppress_small=True)[1:-1]
        send_msg = "s," + send_msg + ","
        conn.send(send_msg.encode())


# Read in the state information sent back by low level controller
def listen2robot(conn):
    state_length = 7 + 7 + 7 + 42
    state_message = str(conn.recv(2048))[2:-2]
    state_str = list(state_message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx + 1 : idx + 1 + state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
        assert len(state_vector) == state_length
    except (ValueError, AssertionError):
        return None

    state = {
        "q": np.asarray(state_vector[0:7]),
        "dq": np.asarray(state_vector[7:14]),
        "tau": np.asarray(state_vector[14:21]),
        "J": np.array(state_vector[21:]).reshape(7, 6).T,
    }

    return state


# Read Full State Info
def readState(conn):
    while True:
        state = listen2robot(conn)
        if state is not None:
            break
    return state


def imitate() -> None:
    # Parse Arguments
    print("[*] Starting up...")
    args = ArgumentParser().parse_args()
    print('\t[*] "I\'m rooting for the machines." (Claude Shannon)')

    # Initialize Model
    print("\n[*] Loading Model...")
    model = Model(args)

    # Connect to Gamepad, Robot
    print("\n[*] Connecting to Gamepad...")
    joystick = JoystickControl()

    print("[*] Launching LibFranka Controller & Connecting to Low-Level Controller...")
    lvc_process = Popen("exec ~/libfranka/build/examples/lilaVelocityController 1", stdout=DEVNULL, shell=True)
    control_conn, gripper_conn = connect2robot(8080), connect2robot(8081)

    # Create Logging Variables
    timestamps, states, instructions, actions, pauses = [], [], [], [], []
    go_home, fire_gripper = [], []
    current_step = 0

    # Get the First Language Instruction to Start the Party!
    start_instruction_time = time.time()
    instruction = input("Enter Instruction =>> ")
    model.embed(instruction)
    instructions.append(
        {
            "Step": current_step,
            "Instruction": instruction,
            "Retrieved Exemplar": model.exemplar,
            "Time to Process": time.time() - start_instruction_time,
        }
    )

    # Drop into Control Loop
    print("\n[*] Starting Execution...")
    current_timestamp, start_time = time.time(), time.time()
    a_continue_execution, press_a, press_a_t, press_b, press_b_t = True, False, time.time(), False, time.time()
    gripper_open = True
    try:
        while True:
            # Read the Robot State
            state = readState(control_conn)

            # Measure Joystick Input
            a, b, x, _, stop = joystick.input()

            # Log State, Zs, Timestamp
            timestamps.append({"Step": current_step, "Timestamp": current_timestamp})
            states.append({"Step": current_step, "State": state})

            # Stop Condition
            if stop:
                print("[*] You pressed <START>, so exiting...")
                break

            # X / GoHome Condition
            elif x:
                # Go Home Timing
                home_start_time = time.time()

                # Kill Existing Sockets
                control_conn.close()
                gripper_conn.close()

                # Kill Libfranka Velocity Controller
                lvc_process.kill()

                # Call Reset to Home Position --> libfranka/build/examples/go_JointPosition robot_ip
                call("~/libfranka/build/examples/go_JointPosition 172.16.0.2", shell=True)

                # Re-initialize LF LILAVelocityController
                lvc_process = Popen(
                    "exec ~/libfranka/build/examples/lilaVelocityController 0", stdout=DEVNULL, shell=True
                )
                control_conn, gripper_conn = connect2robot(8080), connect2robot(8081)

                # Open Gripper if it isn't open already!
                if not gripper_open:
                    # Sleep just in case...
                    time.sleep(1)

                    # Send Fire-Gripper Command
                    send2robot(gripper_conn, qdot=[0.0 for _ in range(args.action_dim)], gripper=True)

                    # Set Gripper Open
                    gripper_open = not gripper_open

                # Log GoHome
                go_home.append({"Step": current_step, "goHome": True, "timeToHome": time.time() - home_start_time})

                # Collect New Instruction
                start_instruction_time = time.time()
                instruction = input("Enter Instruction =>> ")
                model.embed(instruction)
                instructions.append(
                    {
                        "Step": current_step,
                        "Instruction": instruction,
                        "Retrieved Exemplar": model.exemplar,
                        "Time to Process": time.time() - start_instruction_time,
                    }
                )

            # Handle Gripper Logic -- If B-Button Pressed, Fire Gripper
            elif b:
                if not press_b:
                    press_b, press_b_t = True, time.time()

                    # Send Fire-Gripper Command
                    send2robot(gripper_conn, qdot=[0.0 for _ in range(args.action_dim)], gripper=True)

                    # Set Gripper Open
                    gripper_open = not gripper_open

                    # Log Fire Gripper
                    fire_gripper.append({"Step": current_step, "fireGripper": True})

                # Make sure not "holding" button on press!
                if time.time() - press_b_t > 0.25:
                    press_b = False

            # A / Stop Execution
            elif a:
                if not press_a:
                    press_a, press_a_t = True, time.time()

                    # Update Execution (Pause/Play)
                    a_continue_execution = not a_continue_execution

                    # Log Pauses
                    pauses.append({"Step": current_step, "Pause/Play": True})

                # Make sure not "holding" button on press!
                if time.time() - press_a_t > 0.25:
                    press_a = False

            # Otherwise, Imitation Learning
            else:
                if a_continue_execution:
                    # Decode Joint Action, Gripper
                    qdot = [vel * args.action_scale for vel in model.act(state["q"])]

                    # Send Joint-Velocity Command
                    send2robot(control_conn, qdot)

                    # Log Action
                    actions.append({"Step": current_step, "Action": qdot})

            # Loop Invariants
            current_step += 1
            current_timestamp = time.time()

    except (KeyboardInterrupt, ConnectionResetError, BrokenPipeError):
        # Just don't crash the program on Ctrl-C or Socket Error (Controller Death)
        print("\n[*] Terminating...")

    finally:
        print("\n[*] Cleaning up...")

        # Dump Logs!
        os.makedirs(f"logs/{args.name}/", exist_ok=True)
        with open(
            f"logs/{args.name}/study-control=imitation-{args.mode}-task={args.task_idx}-trial={args.trial_idx}.pkl",
            "wb",
        ) as f:
            pickle.dump(
                {
                    "Timestamps": timestamps,
                    "States": states,
                    "Instructions": instructions,
                    "Pauses": pauses,
                    "Actions": actions,
                    "Home Actions": go_home,
                    "Gripper Actions": fire_gripper,
                    "Total Time": time.time() - start_time,
                },
                f,
            )

        # Connection Cleanup
        control_conn.close()
        gripper_conn.close()
        lvc_process.kill()


if __name__ == "__main__":
    imitate()
