# Mjolnir Architecture

It's always a good idea to write up how a codebase is structured -- I generally like to talk about code flow,
entry points, and key APIs. For example, a top level `train.py`, separate folders for evaluation/visualization, a
centralized `src` directory with separate modules for preprocessing, model definition (with Lightning, this is all
you need for training), and other utilities.

For more complex repositories, this file should have structure about not only how to read the codebase, but also how
to build on top of it -- for example, if I wanted to add a new Simulation environment, a rough sketch of the flow to
do that, and so on.

Try keeping this up to date, but not until after code has relatively stabilized.
