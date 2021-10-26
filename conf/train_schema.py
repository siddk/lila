"""
schema.py

Cerberus Schema used by Quinine for `train.py`. Note - if multiple entry points (with different configs), then write
separate schemas for each.
"""
from typing import Any, Dict

from quinine.common.cerberus import default, merge, nullable, tinteger, tstring


def get_train_schema() -> Dict[str, Any]:
    """ Get the Cerberus Schema for the Quinine Config used by `train.py`. """

    # Update as Necessary --> see `https://github.com/krandiash/quinine#cerberus-schemas-for-validation`
    schema = {"run_id": merge(tstring, nullable, default(None)), "seed": merge(tinteger, nullable, default(21))}

    return schema
