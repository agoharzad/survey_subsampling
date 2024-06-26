#!/usr/bin/env python
"""Converter script for Rdata format input data."""
from argparse import ArgumentParser

from pyreadr import read_r


def run() -> None:
    """Runscript to convert Rdata."""
    # TODO: improve docstrings, helptext, and the like
    parser = ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    results = parser.parse_args()

    # Load file
    datafile = read_r(results.infile)

    # Convert to dataframe
    df = datafile[None]

    # Write it out to parquet
    df.to_parquet(results.outfile)


if __name__ == "__main__":
    run()
