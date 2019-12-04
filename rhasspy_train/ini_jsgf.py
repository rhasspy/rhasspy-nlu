"""Parsing code for ini/JSGF grammars."""
import io
import configparser
import logging
import re
import typing
from collections import defaultdict

import attr

from .jsgf import Rule, Sentence

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
                grammar.rules.append(Rule.parse(line))

        return grammar


class IniJsgf:
    """Parser for JSGF embedded inside an ini file."""

    @classmethod
    def parse(
        cls, source: typing.TextIO, whitelist: typing.Optional[typing.List[str]] = None
    ) -> typing.Dict[str, typing.List[typing.Union[Sentence, Rule]]]:
        """Parse multiple JSGF grammars from an ini file."""
        # Create ini parser
        config = configparser.ConfigParser(
            allow_no_value=True, strict=False, delimiters=["="]
        )

        config.optionxform = str  # case sensitive
        config.read_file(source)

        _LOGGER.debug("Loaded ini file")

        # Process configuration sections
        sentences: typing.Dict[
            str, typing.List[typing.Union[Sentence, Rule]]
        ] = defaultdict(list)

        # Parse each section (intent)
        for sec_name in config.sections():
            # Exclude if not in whitelist.
            # Empty whitelist means keep all.
            if (whitelist is not None) and (sec_name not in whitelist):
                logger.debug("Skipping %s (not in whitelist)", sec_name)
                continue

            # Processs settings (sentences/rules)
            for k, v in config[sec_name].items():
                if v is None:
                    # Collect non-valued keys as sentences
                    sentence = k.strip()

                    # Fix \[ escape sequence
                    sentence = re.sub(r"\\\[", "[", sentence)

                    sentences[sec_name].append(Sentence.parse(sentence))
                else:
                    # Collect key/value pairs as JSGF rules
                    rule = "<{0}> = ({1});".format(k.strip(), v.strip())

                    # Fix \[ escape sequence
                    rule = re.sub(r"\\\[", "[", rule)

                    sentences[sec_name].append(Rule.parse(rule))

        return sentences
