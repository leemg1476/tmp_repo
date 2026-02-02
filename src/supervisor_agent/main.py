"""CLI entrypoint for the supervisor agent."""
from __future__ import annotations

import argparse
from pprint import pprint

from .graph import run_supervisor
from .config import configure_observability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervisor agent once.")
    parser.add_argument("query", help="User query to handle.")
    parser.add_argument(
        "--mode",
        choices=["serial", "parallel"],
        help="Execution mode for tasks (serial or parallel).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_observability()
    result = run_supervisor(args.query, execution_mode=args.mode)
    pprint(result)


if __name__ == "__main__":
    main()
