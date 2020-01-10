"""Parsing code for ini/JSGF grammars."""
import io
import configparser
import logging
from pathlib import Path
import re
import typing
from collections import defaultdict

import attr

from .jsgf import Expression, Rule, Sentence, get_expression_count

_LOGGER = logging.getLogger(__name__)


@attr.s
class Grammar:
    """Named JSGF grammar with rules."""

    grammar_name: str = attr.ib(default="")
    rules: typing.List[Rule] = attr.ib(factory=list)

    GRAMMAR_DECLARATION = re.compile(r"^grammar ([^;]+);$")

    @classmethod
    def parse(cls, source: typing.TextIO) -> "Grammar":
        """Parse single JSGF grammar."""
        grammar = Grammar()

        # Read line-by-line
        for line in source:
            line = line.strip()
            if line.startswith("#") or (not line):
                # Skip comments/blank lines
                continue

            grammar_match = Grammar.GRAMMAR_DECLARATION.match(line)
            if grammar_match is not None:
                # grammar GrammarName;
                grammar.grammar_name = grammar_match.group(1)
            else:
                # public <RuleName> = rule body;
                # <RuleName> = rule body;
                grammar.rules.append(Rule.parse(line))  # pylint: disable=E1101

        return grammar


# -----------------------------------------------------------------------------


def parse_ini(
    source: typing.Union[str, Path, typing.TextIO],
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    sentence_transform: typing.Callable[[str], str] = None,
) -> typing.Dict[str, typing.List[typing.Union[Sentence, Rule]]]:
    """Parse multiple JSGF grammars from an ini file."""
    intent_filter = intent_filter or (lambda x: True)
    if isinstance(source, str):
        source = io.StringIO(source)
    elif isinstance(source, Path):
        source = open(source, "r")

    # Process configuration sections
    sentences: typing.Dict[
        str, typing.List[typing.Union[Sentence, Rule]]
    ] = defaultdict(list)

    try:
        # Create ini parser
        config = configparser.ConfigParser(
            allow_no_value=True, strict=False, delimiters=["="]
        )

        # case sensitive
        config.optionxform = str  # type: ignore
        config.read_file(source)

        _LOGGER.debug("Loaded ini file")

        # Parse each section (intent)
        for sec_name in config.sections():
            # Exclude if filtered out.
            if not intent_filter(sec_name):
                _LOGGER.debug("Skipping %s", sec_name)
                continue

            # Processs settings (sentences/rules)
            for k, v in config[sec_name].items():
                if v is None:
                    # Collect non-valued keys as sentences
                    sentence = k.strip()

                    # Fix \[ escape sequence
                    sentence = sentence.replace("\\[", "[")

                    if sentence_transform:
                        # Do transform
                        sentence = sentence_transform(sentence)

                    sentences[sec_name].append(Sentence.parse(sentence))
                else:
                    sentence = v.strip()

                    if sentence_transform:
                        # Do transform
                        sentence = sentence_transform(sentence)

                    # Collect key/value pairs as JSGF rules
                    rule = f"<{k.strip()}> = ({sentence});"

                    # Fix \[ escape sequence
                    rule = rule.replace("\\[", "[")

                    sentences[sec_name].append(Rule.parse(rule))
    finally:
        source.close()

    return sentences


# -----------------------------------------------------------------------------


def split_rules(
    intents: typing.Dict[str, typing.List[typing.Union[Sentence, Rule]]],
    replacements: typing.Optional[typing.Dict[str, typing.Iterable[Sentence]]] = None,
) -> typing.Tuple[
    typing.Dict[str, typing.List[Sentence]], typing.Dict[str, typing.Iterable[Sentence]]
]:
    """Seperate out rules and sentences from all intents."""
    sentences: typing.Dict[str, typing.List[Sentence]] = {}
    replacements = replacements or {}

    for intent_name, intent_exprs in intents.items():
        sentences[intent_name] = []

        # Extract rules and fold them into replacements
        for expr in intent_exprs:
            if isinstance(expr, Rule):
                # Rule
                rule_name = expr.rule_name
                if "." not in rule_name:
                    # Add local replacement
                    replacements[f"<{rule_name}>"] = [expr.rule_body]

                    # Use fully qualified name too
                    rule_name = f"{intent_name}.{rule_name}"

                # Surround with <>
                rule_name = f"<{rule_name}>"
                replacements[rule_name] = [expr.rule_body]
            else:
                sentences[intent_name].append(expr)

    return sentences, replacements


# -----------------------------------------------------------------------------


def get_intent_counts(
    intents: typing.Dict[str, typing.List[typing.Union[Sentence, Rule]]],
    replacements: typing.Optional[typing.Dict[str, typing.Iterable[Sentence]]] = None,
    exclude_slots: bool = True,
    count_dict: typing.Optional[typing.Dict[Expression, int]] = None,
):
    """Get number of possible sentences for each intent."""
    sentences, replacements = split_rules(intents, replacements)
    intent_counts: typing.Dict[str, int] = defaultdict(int)

    for intent_name, intent_sentences in sentences.items():
        # Compute counts for all sentences
        intent_counts[intent_name] = max(
            1,
            sum(
                get_expression_count(
                    s, replacements, exclude_slots=exclude_slots, count_dict=count_dict
                )
                for s in intent_sentences
            ),
        )

    return intent_counts
