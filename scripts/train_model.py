#!/usr/bin/env python
import logging
import sys
from pathlib import Path

import click
from IPython.core import ultratb

import mi_agent

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)
# turn UserWarning messages to errors to find the actual cause
# import warnings
# warnings.simplefilter("error")

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="path to config file",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.version_option(mi_agent.__version__)
def main(cfg_path: Path, log_level: int):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # YOUR CODE GOES HERE! Keep the main functionality in src/mi_agent
    # est = mi_agent.models.Estimator()


if __name__ == "__main__":
    main()
