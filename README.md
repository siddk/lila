# LILA

> *LILA*: Language-Informed Latent Actions

Code and Experiments for Language-Informed Latent Actions (LILA), for using natural language to guide assistive
teleoperation.

This code bundles code that can be deployed on a Franka Emika Panda Arm, including utilities for processing
collected demonstrations (you can find our actual demo data in the `data/` directory!), training various LILA and
Imitation Learning models, and running live studies.

---

## Quickstart

Assumes `lila` is the current working directory! This repository also comes with out-of-the-box linting and strict
pre-commit checking... should you wish to *turn off this functionality* you can omit the `pre-commit install` lines
below. If you do choose to use these features, you can run `make autoformat` to automatically clean code, and `make
check` to identify any violations.

## Repository Structure

High-level overview of repository file-tree:

+ `conf` - Quinine Configurations (`.yaml`) for various runs (used in lieu of `argparse` or `typed-argument-parser`)
+ `environments` - Serialized Conda Environments for running on CPU. Other architectures/CUDA toolkit
environments can be added here as necessary.
+ `robot/` - Core `libfranka` robot control code -- simple joint velocity controll w/ Gripper control.
+ `src/` - Source Code - has all utilities for preprocessing, Lightning Model definitions, utilities.
    + `preprocessing/` - Preprocessing Code for creating Torch Datasets for Training LILA/Imitation Models.
    + `models/` - Lightning Modules for LILA-FiLM and Imitation-FiLM Architectures.
+ `train.py` - Top-Level (main) entry point to repository, for training and evaluating models. Run this first, pointing
it at the appropriate configuration in `conf/`!.
+ `Makefile` - Top-level Makefile (by default, supports `conda` serialization, and linting). Expand to your needs.
+ `.flake8` - Flake8 Configuration File (Sane Defaults).
+ `.pre-commit-config.yaml` - Pre-Commit Configuration File (Sane Defaults).
+ `pyproject.toml` - Black and isort Configuration File (Sane Defaults).+ `README.md` - You are here!
+ `README.md` - You are here!
+ `LICENSE` - By default, research code is made available under the MIT License.

### Local Development - CPU (Mac OS & Linux)

Note: Assumes that `conda` (Miniconda or Anaconda are both fine) is installed and on your path. Use the `-cpu`
environment file.

```bash
conda env create -f environments/environment-cpu.yaml
conda activate lila
pre-commit install
```

### GPU Development - Linux w/ CUDA 11.0

```bash
conda env create -f environments/environment-gpu.yaml  # Choose CUDA Kernel based on Hardware - by default used 11.0!
conda activate lila
pre-commit install
```

Note: This codebase should work naively for all PyTorch > 1.7, and any CUDA version; if you run into trouble building
this repository, please file an issue!

---

## Training LILA or Imitation Models

To train models using the already collected demonstrations.

```
# LILA
python train.py --config conf/lila-config.yaml

# No-Language Latent Actions
python train.py --config conf/no-lang-config.yaml

# Imitatation Learning (Behavioral Cloning w/ DART-style Augmentation)
python train.py --config conf/imitation-config.yaml
```

This will dump models to `runs/{lila-final, no-lang-final, imitation-final}/`. These paths are **hard-coded** in the
respective teleoperation/execution files below; if you change these paths, be sure to change the below files as well!

## Teleoperating with LILA or End-Effector Control

First, make sure to add the custom Velocity Controller written for the Franka Emika Panda Robot Arm (written using
Libfranka) to ~/libfranka/examples on your robot control box. The controller can be found in
`robot/libfranka/lilaVelocityController.cpp`.

Then, make sure to update the path of the model trained in the previous step (for LILA) in `teleoperate.py`. Finally,
you can drop into controlling the robot with a LILA model (and Joystick - make sure it's plugged in!) with:

```
# LILA Control
python teleoperate.py

# For No-Language Control, just change the arch!
python teleoperate.py --arch no-lang

# Pure End-Effector Control is also implemented by Default
python teleoperate.py --arch endeff
```

## Running Imitation Learning

Add the Velocity Controller as described above. Then, make sure to update the path to the trained model in `imitate.py`
and run the following:

```
python imitate.py
```

---

### Collecting Kinesthetic Demonstrations

Each lab (and corresponding robot) is built with a different stack, and different preferred ways of recording
Kinesthetic demonstrations. We have a rudimentary script `record.py` that shows how we do this using sockets, and the
default `libfranka readState.cpp` built-in script. This script dumps demonstrations that can be immediately used to train
latent action models.

### Start-Up from Scratch

In case the above `conda` environment loading does not work for you, here are the concrete package dependencies
required to run LILA:

```bash
conda create --name lila python=3.8
conda activate lila
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython jupyter
conda install pytorch-lightning -c conda-forge

pip install black flake8 isort matplotlib pre-commit pygame quinine transformers typed-argument-parser wandb
```
