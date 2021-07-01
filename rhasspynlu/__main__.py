"""Command-line utility for rhasspynlu"""
import argparse
import io
import json
import logging
import sys
import typing
from pathlib import Path

from .fsticuffs import recognize
from .ini_jsgf import Expression, Word, parse_ini, split_rules
from .intent import Recognition
from .jsgf import walk_expression
from .jsgf_graph import sentences_to_graph, graph_to_json, graph_to_gzip_pickle
from .numbers import number_range_transform, number_transform
from .slots import get_slot_replacements

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method"""
    parser = argparse.ArgumentParser("rhasspynlu")
    parser.add_argument(
        "--sentences", required=True, action="append", help="Sentences ini files"
    )
    parser.add_argument(
        "--slot-dir", action="append", default=[], help="Directory with slot files"
    )
    parser.add_argument(
        "--slot-programs-dir",
        action="append",
        default=[],
        help="Directory with slot programs",
    )
    parser.add_argument(
        "--casing",
        default="keep",
        choices=["keep", "lower", "upper"],
        help="Casing transformation (default: keep)",
    )
    parser.add_argument(
        "--no-replace-numbers",
        action="store_true",
        help="Don't expand numbers into words",
    )
    parser.add_argument(
        "--number-language", default="en", help="Language used for num2words"
    )
    parser.add_argument(
        "--write-json", action="store_true", help="Write JSON graph to stdout and exit"
    )
    parser.add_argument(
        "--write-pickle",
        action="store_true",
        help="Write gzipped graph pickle to stdout and exit",
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

    args.slot_dir = [Path(p) for p in args.slot_dir]
    args.slot_programs_dir = [Path(p) for p in args.slot_programs_dir]

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

    _LOGGER.debug("Processing sentences")
    sentences, replacements = split_rules(intents)

    # Transform words
    word_transform: typing.Optional[typing.Callable[[str], str]] = None

    if args.casing == "lower":
        word_transform = str.lower
    elif args.casing == "upper":
        word_transform = str.upper

    word_visitor: typing.Optional[
        typing.Callable[[Expression], typing.Union[bool, Expression]]
    ] = None

    if word_transform:
        # Apply transformation to words

        def transform_visitor(word: Expression):
            if isinstance(word, Word):
                assert word_transform
                new_text = word_transform(word.text)

                # Preserve case by using original text as substition
                if (word.substitution is None) and (new_text != word.text):
                    word.substitution = word.text

                word.text = new_text

            return word

        word_visitor = transform_visitor

    # Apply case/number transforms
    if word_visitor or (not args.no_replace_numbers):
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                if not args.no_replace_numbers:
                    # Replace number ranges with slot references
                    # type: ignore
                    walk_expression(sentence, number_range_transform, replacements)

                if word_visitor:
                    # Do case transformation
                    # type: ignore
                    walk_expression(sentence, word_visitor, replacements)

    # Load slot values
    slot_replacements = get_slot_replacements(
        intents,
        slots_dirs=args.slot_dir,
        slot_programs_dirs=args.slot_programs_dir,
        slot_visitor=word_visitor,
    )

    # Merge with existing replacements
    for slot_key, slot_values in slot_replacements.items():
        replacements[slot_key] = slot_values

    if not args.no_replace_numbers:
        # Do single number transformations
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                walk_expression(
                    sentence,
                    lambda w: number_transform(w, args.number_language),
                    replacements,
                )

    # Convert to directed graph
    _LOGGER.debug("Converting to graph")
    graph = sentences_to_graph(sentences, replacements=replacements)

    if args.write_json:
        json.dump(graph_to_json(graph), sys.stdout)
        return

    if args.write_pickle:
        graph_to_gzip_pickle(graph, sys.stdout.buffer, filename="intent_graph.pickle")
        return

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
