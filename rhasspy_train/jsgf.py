"""Parses a subset of JSGF into objects."""
import re
import typing
from enum import Enum

import attr


@attr.s
class Substitutable:
    """Indicates an expression may be replaced with some text."""

    substitution: typing.Optional[str] = attr.ib(default=None)


@attr.s
class Tag(Substitutable):
    """{tag} attached to an expression."""

    tag_text: str = attr.ib(default="")


@attr.s
class Taggable:
    """Indicates an expression may be tagged."""

    tag: typing.Optional[Tag] = attr.ib(default=None)


@attr.s
class Expression:
    """Base class for most JSGF types."""

    text: str = attr.ib(default="")


@attr.s
class Word(Expression, Taggable, Substitutable):
    """Single word/token."""

    pass


class SequenceType(str, Enum):
    """Type of a sequence. Optionals are alternatives with an empty option."""

    GROUP = "group"
    ALTERNATIVE = "alternative"


@attr.s
class Sequence(Expression, Taggable, Substitutable):
    """Ordered sequence of expressions. Supports groups, optionals, and alternatives."""

    items: typing.List[Expression] = attr.ib(factory=list)
    type: SequenceType = attr.ib(default=SequenceType.GROUP)


@attr.s
class RuleReference(Expression, Taggable):
    """Reference to a rule by <name> or <grammar.name>."""

    rule_name: str = attr.ib(default="")
    grammar_name: typing.Optional[str] = attr.ib(default=None)


@attr.s
class SlotReference(Expression, Taggable):
    """Reference to a slot by $name."""

    slot_name: str = attr.ib(default="")


@attr.s
class Sentence(Sequence):
    """Sequence representing a complete sentence template."""

    @classmethod
    def parse(cls, text: str) -> "Sentence":
        s = Sentence(text=text)
        parse_expression(s, text)

        # Unwrap single child
        if (len(s.items) == 1) and isinstance(s.items[0], Sequence):
            item = s.items[0]
            s.type = item.type
            s.text = item.text
            s.items = item.items
            s.tag = item.tag
            s.substitution = item.substitution

        return s


@attr.s
class Rule:
    """Named rule with body."""

    RULE_DEFINITION = re.compile(r"^(public)?\s*<([^>]+)>\s*=\s*([^;]+)(;)?$")

    rule_name: str = attr.ib()
    rule_body: Sentence = attr.ib()
    public: bool = attr.ib(default=False)
    text: str = attr.ib(default="")

    @classmethod
    def parse(cls, text: str) -> "Rule":
        # public <RuleName> = rule body;
        # <RuleName> = rule body;
        rule_match = Rule.RULE_DEFINITION.match(text)
        assert rule_match is not None

        public = rule_match.group(1) is not None
        rule_name = rule_match.group(2)
        rule_text = rule_match.group(3)

        s = Sentence.parse(rule_text)
        return Rule(rule_name=rule_name, rule_body=s, public=public, text=text)


# -----------------------------------------------------------------------------


def split_words(text: str) -> typing.Iterable[Expression]:
    """Split words by whitespace. Detect slot references and substitutions."""
    for token in re.split(r"\s+", text):
        if token.startswith("$"):
            yield SlotReference(text=token, slot_name=token[1:])
        elif ":" in token:
            lhs, rhs = token.split(":", maxsplit=1)
            yield Word(text=lhs, substitution=rhs)
        else:
            yield Word(text=token)


def parse_expression(
    root: typing.Optional[Sequence],
    text: str,
    end: typing.Optional[str] = None,
    is_literal=True,
) -> typing.Optional[int]:
    """Parse a full expression. Return index in text where current expression ends."""
    found: bool = False
    next_index: int = 0
    literal: str = ""
    last_taggable: typing.Optional[Taggable] = None

    # Process text character-by-character
    for current_index, c in enumerate(text):
        if current_index < next_index:
            # Skip ahread
            current_index += 1
            continue

        if current_index > 0:
            last_c = text[current_index - 1]
        else:
            last_c = ""

        next_index = current_index + 1

        if c == end:
            # Found end character of expression (e.g., ])
            next_index += 1
            found = True
            break
        elif (c == ":") and (last_c in [")", "]"]):
            # Handle sequence substitution
            assert isinstance(last_taggable, Substitutable)
            next_index = parse_expression(
                None, text[current_index + 1 :], end=" ", is_literal=False
            )

            if next_index is None:
                # End of text
                next_index = len(text) + 1
            else:
                next_index += current_index

            last_taggable.substitution = text[current_index + 1 : next_index - 1]

        elif c in ["<", "(", "[", "{", "|"]:
            # Begin group/tag/alt/etc.

            # Break literal here
            literal = literal.strip()
            if literal:
                assert root is not None
                words = list(split_words(literal))
                root.items.extend(words)
                last_taggable = words[-1]
                literal = ""

            if c == "<":
                # Rule reference
                assert root is not None
                rule = RuleReference()
                next_index = current_index + parse_expression(
                    None, text[current_index + 1 :], end=">", is_literal=False
                )


                rule_name = text[current_index + 1 : next_index - 1]
                if "." in rule_name:
                    # Split by last dot
                    last_dot = rule_name.rindex(".")
                    rule.grammar_name = rule_name[:last_dot]
                    rule.rule_name = rule_name[last_dot + 1 :]
                else:
                    # Use entire name
                    rule.rule_name = rule_name

                rule.text = text[current_index:next_index]
                root.items.append(rule)
                last_taggable = rule
            elif c == "(":
                # Group (expression)
                assert root is not None
                group = Sequence(type=SequenceType.GROUP)
                next_index = current_index + parse_expression(
                    group, text[current_index + 1 :], end=")"
                )

                group.text = text[current_index + 1 : next_index - 1]
                root.items.append(group)
                last_taggable = group
            elif c == "[":
                # Optional
                optional = Sequence(type=SequenceType.ALTERNATIVE)
                next_index = current_index + parse_expression(
                    optional, text[current_index + 1 :], end="]"
                )

                # Empty alternative
                optional.items.append(Word(text=""))
                optional.text = text[current_index + 1 : next_index - 1]
                root.items.append(optional)
                last_taggable = optional
            elif c == "{":
                assert last_taggable is not None
                tag = Tag()

                # Tag
                next_index = current_index + parse_expression(
                    None, text[current_index + 1 :], end="}", is_literal=False
                )

                # Exclude {}
                tag.tag_text = text[current_index + 1 : next_index - 1]

                if ":" in tag.tag_text:
                    lhs, rhs = tag.tag_text.split(":", maxsplit=1)
                    tag.tag_text = lhs
                    tag.substitution = rhs

                last_taggable.tag = tag
            elif c == "|":
                assert root is not None
                root.type = SequenceType.ALTERNATIVE
        else:
            # Accumulate into current literal
            literal += c

    # End of expression; Break literal.
    literal = literal.strip()
    if is_literal and literal:
        assert root is not None
        words = list(split_words(literal))
        root.items.extend(words)

    if (end is not None) and (not found):
        # Signal end not found
        return None

    return next_index
