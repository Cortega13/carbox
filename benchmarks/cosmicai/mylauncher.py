"""Launch pylauncher with a commandlines file."""

import argparse
from pathlib import Path

import pylauncher


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch MPI commandlines with pylauncher")
    parser.add_argument(
        "--command-file",
        type=Path,
        required=True,
        help="Path to commandlines.txt",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default="job",
        help="pylauncher debug mode",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    pylauncher.ClassicLauncher(str(args.command_file), debug=args.debug)


if __name__ == "__main__":
    main()
