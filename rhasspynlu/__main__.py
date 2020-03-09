"""Command-line utility for rhasspynlu"""
import argparse
import io
import json
import logging
import os
import sys

from . import graph_to_fst, graph_to_json, intents_to_graph, parse_ini
from .fsticuffs import sample_by_intent

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method"""
    parser = argparse.ArgumentParser("rhasspynlu")
    parser.add_argument("sentences_file", nargs="*", help="Sentences ini files")
    parser.add_argument(
        "--fst", action="store_true", help="Output FST text instead of JSON"
    )
    parser.add_argument("--fst-text", default="fst.txt", help="Path to FST text file")
    parser.add_argument(
        "--fst-isymbols",
        default="fst.isymbols.txt",
        help="Path to FST input symbols file",
    )
    parser.add_argument(
        "--fst-osymbols",
        default="fst.osymbols.txt",
        help="Path to FST output symbols file",
    )
    parser.add_argument(
        "--sample", type=int, help="Generate sample sentences (0 = all)"
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

    # Read sentences ini from files or stdin
    try:
        if args.sentences_file:
            with io.StringIO() as ini_file:
                for sentences_path in args.sentences_file:
                    _LOGGER.debug("Reading %s", sentences_path)

                    with open(sentences_path, "r") as sentences_file:
                        ini_file.write(sentences_file.read())
                        print("", file=ini_file)

                ini_text = ini_file.getvalue()
        else:
            if os.isatty(sys.stdin.fileno()):
                print("Reading sentences ini from stdin...", file=sys.stderr)

            ini_text = sys.stdin.read()

        _LOGGER.debug("Parsing sentences")
        intents = parse_ini(ini_text)

        _LOGGER.debug("Converting to graph")
        graph = intents_to_graph(intents)

        if args.fst:
            _LOGGER.debug("Converting to FST")
            graph_fst = graph_to_fst(graph)

            _LOGGER.debug(
                "Writing %s, %s, %s",
                args.fst_text,
                args.fst_isymbols,
                args.fst_osymbols,
            )

            graph_fst.write_fst(args.fst_text, args.fst_isymbols, args.fst_osymbols)
        elif args.sample is not None:
            if args.sample > 0:
                num_samples = args.sample
                _LOGGER.debug("Generating %s sample(s)", args.sample)
            else:
                num_samples = None
                _LOGGER.debug("Generating all sentences")

            sentences_by_intent = sample_by_intent(graph, num_samples)

            # Output JSON
            json.dump(
                {
                    intent: [r.asdict() for r in recognitions]
                    for intent, recognitions in sentences_by_intent.items()
                },
                sys.stdout,
            )
        else:
            # Output JSON
            _LOGGER.debug("Writing to stdout")
            graph_dict = graph_to_json(graph)

            json.dump(graph_dict, sys.stdout)
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
