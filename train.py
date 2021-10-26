"""
train.py

Core training script -- loads and preprocesses, instantiates a Lightning Module, and runs training. Fill in with more
repository/project-specific training details!

Run with: `python train.py --config conf/config.yaml`
"""
from datetime import datetime

from quinine import QuinineArgumentParser

from conf.train_schema import get_train_schema


def train() -> None:
    # Parse Quinfig (via Quinine Argparse Binding)
    print("[*] Mjolnir :: Launching =>>>")
    quinfig = QuinineArgumentParser(schema=get_train_schema()).parse_quinfig()
    print('\t=>> "Thunder is good, thunder is impressive; but it is Lightning that does all the work" (Mark Twain)')

    # Create Unique Run Name
    run_id = quinfig.run_id
    if run_id is None:
        run_id = f"mjolnir+{datetime.now().strftime('%Y-%m-%d-%H:%M')}"

    # Do stuff...


if __name__ == "__main__":
    train()
