from __future__ import annotations

import argparse
from typing import Sequence


def float_or_int(s: str) -> float | int:
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse_args(*extra_flag: str, args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--shared-config", default="configs/shared.yaml")
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--domain", type=int, required=True)
    parser.add_argument("--ratio", type=float_or_int, required=True)
    parser.add_argument("--decay", help="ema decay", type=float, default=0.99)

    for flag in extra_flag:
        parser.add_argument(flag, action="store_true")
    return parser.parse_args(args)
