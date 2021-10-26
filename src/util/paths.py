"""
paths.py

Utility function for initializing the appropriate directories/sub-directories on the start of each run. Decoupled from
main code in case we want separate directory structures/artifact storage based on infrastructure.
"""
from pathlib import Path
from typing import Dict


def create_paths(run_id: str, run_dir: str = "runs/") -> Dict[str, Path]:
    """
    Create the necessary directories and sub-directories conditioned on the `run_id` and run directory.

    :param run_id: Unique Run Identifier.
    :param run_dir: Path to run directory to save model checkpoints and run metrics.
    """
    paths = {
        # Top-Level Directory for a Given Run
        "runs": Path(run_dir, run_id)
    }

    # Programatically Create Paths for each Directory
    for p in paths:
        paths[p].mkdir(parents=True, exist_ok=True)

    return paths
