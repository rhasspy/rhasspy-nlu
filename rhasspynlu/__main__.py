"""Command-line utility for rhasspynlu"""
import argparse
import io
import json
import logging
import os
import sys

from . import intents_to_graph, parse_ini
from .fsticuffs import recognize
from .intent import Recognition

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method"""
    parser = argparse.ArgumentParser("rhasspynlu")
    parser.add_argument(
        "--sentences", required=True, action="append", help="Sentences ini files"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Read sentences
    with io.StringIO() as ini_file:
        for sentences_path in args.sentences:
            _LOGGER.debug("Reading %s", sentences_path)

            with open(sentences_path, "r") as sentences_file:
                ini_file.write(sentences_file.read())
                print("", file=ini_file)

        ini_text = ini_file.getvalue()

    _LOGGER.debug("Parsing sentences")
    intents = parse_ini(ini_text)

    _LOGGER.debug("Converting to graph")
    graph = intents_to_graph(intents)

    # Read sentences stdin
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            results = recognize(line, graph)
            if results:
                result = results[0]
            else:
                result = Recognition.empty()

            json.dump(result.asdict(), sys.stdout)
            print("", flush=True)

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
