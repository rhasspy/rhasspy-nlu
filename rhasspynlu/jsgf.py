"""Parses a subset of JSGF into objects."""
import re
import typing
from enum import Enum

import attr


@attr.s
class Substitutable:
    """Indicates an expression may be replaced with some text."""

    # Replacement text
    substitution: typing.Optional[str] = attr.ib(default=None)

    # Names of converters to apply after substitution
    converters: typing.List[str] = attr.ib(factory=list)


@attr.s
class Tag(Substitutable):
    """{tag} attached to an expression."""

    # Name of tag (entity)
    tag_text: str = attr.ib(default="")


@attr.s
class Taggable:
    """Indicates an expression may be tagged."""

    # Tag to be applied
    tag: typing.Optional[Tag] = attr.ib(default=None)


@attr.s
class Expression:
    """Base class for most JSGF types."""

    # Text representation expression
    text: str = attr.ib(default="")


@attr.s
class Word(Expression, Taggable, Substitutable):
    """Single word/token."""

    pass


class SequenceType(str, Enum):
    """Type of a sequence. Optionals are alternatives with an empty option."""

    # Sequence of expressions
    GROUP = "group"

    # Expressions where only one will be recognized
    ALTERNATIVE = "alternative"


@attr.s
class Sequence(Expression, Taggable, Substitutable):
    """Ordered sequence of expressions. Supports groups, optionals, and alternatives."""

    # Items in the sequence
    items: typing.List[Expression] = attr.ib(factory=list)

    # Group or alternative
    type: SequenceType = attr.ib(default=SequenceType.GROUP)


@attr.s
class RuleReference(Expression, Taggable):
    """Reference to a rule by <name> or <grammar.name>."""

    # Name of referenced rule
    rule_name: str = attr.ib(default="")

    # Grammar name of referenced rule
    grammar_name: typing.Optional[str] = attr.ib(default=None)


@attr.s
class SlotReference(Expression, Taggable, Substitutable):
    """Reference to a slot by $name."""

    # Name of referenced slot
    slot_name: str = attr.ib(default="")


@attr.s
class Sentence(Sequence):
    """Sequence representing a complete sentence template."""

    @classmethod
    def parse(cls, text: str) -> "Sentence":
        """Parse a single sentence."""
        s = Sentence(text=text)
        parse_expression(s, text)
        seq = unwrap_sequence(s)
        return Sentence(
            text=seq.text,
            items=seq.items,
            type=seq.type,
            tag=seq.tag,
            substitution=seq.substitution,
        )


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
        """Parse a single rule."""
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


def walk_expression(
    expression: Expression,
    visit: typing.Callable[[Expression], bool],
    replacements: typing.Optional[typing.Dict[str, typing.Iterable[Expression]]] = None,
):
    """Recursively visit nodes in expression."""
    result = visit(expression)

    # Break on explicit False
    if (result is not None) and (not result):
        return

    if isinstance(expression, Sequence):
        for item in expression.items:
            walk_expression(item, visit, replacements)
    elif isinstance(expression, Rule):
        walk_expression(expression.rule_body, visit, replacements)
    elif isinstance(expression, RuleReference):
        key = f"<{expression.rule_name}>"
        assert replacements, key
        for item in replacements[key]:
            walk_expression(item, visit, replacements)
    elif isinstance(expression, SlotReference):
        key = f"${expression.slot_name}"
        assert replacements, key
        for item in replacements[key]:
            walk_expression(item, visit, replacements)


# -----------------------------------------------------------------------------


def split_words(text: str) -> typing.Iterable[Expression]:
    """Split words by whitespace. Detect slot references and substitutions."""
    for token in re.split(r"\s+", text):
        if token.startswith("$"):
            if ":" in token:
                # Slot with substitutions
                lhs, rhs = token[1:].split(":", maxsplit=1)
                yield SlotReference(text=token, slot_name=lhs, substitution=rhs)
            else:
                # Slot without substitutions
                yield SlotReference(text=token, slot_name=token[1:])
        elif ":" in token or "!" in token:
            word = Word(text=token)

            if "!" in token:
                # Word with converter(s)
                # e.g., twenty:20!int
                parts = token.split("!")
                word.text = parts[0]
                word.converters = parts[1:]

            if ":" in word.text:
                # Word with substitution
                # e.g., twenty:20
                lhs, rhs = word.text.split(":", maxsplit=1)
                word.text = lhs
                word.substitution = rhs

            yield word
        else:
            # With without substitution
            yield Word(text=token)


def unwrap_sequence(seq: Sequence) -> Sequence:
    """Recursively unpack sequences with single items."""
    # Unwrap single child
    while (len(seq.items) == 1) and isinstance(seq.items[0], Sequence):
        item = seq.items[0]
        seq.type = item.type
        seq.text = item.text or seq.text
        seq.items = item.items
        seq.tag = item.tag or seq.tag
        seq.substitution = item.substitution or seq.substitution

    return seq


def parse_expression(
    root: typing.Optional[Sequence],
    text: str,
    end: typing.List[str] = None,
    is_literal=True,
) -> typing.Optional[int]:
    """Parse a full expression. Return index in text where current expression ends."""
    end = end or []
    found: bool = False
    next_index: int = 0
    literal: str = ""
    last_taggable: typing.Optional[Taggable] = None
    last_group: typing.Optional[Sequence] = root

    # Process text character-by-character
    for current_index, c in enumerate(text):
        if current_index < next_index:
            # Skip ahread
            current_index += 1
            continue

        # Get previous character
        if current_index > 0:
            last_c = text[current_index - 1]
        else:
            last_c = ""

        next_index = current_index + 1

        if c in end:
            # Found end character of expression (e.g., ])
            next_index += 1
            found = True
            break

        if (c == ":") and (last_c in [")", "]"]):
            # Handle sequence substitution
            assert isinstance(last_taggable, Substitutable)
            next_index = parse_expression(
                None, text[current_index + 1 :], end=[" "] + end, is_literal=False
            )

            if next_index is None:
                # End of text
                next_index = len(text) + 1
            else:
                next_index += current_index - 1

            sub_text = text[current_index + 1 : next_index].strip()
            if "!" in sub_text:
                # Extract converter(s)
                sub_text, *converters = sub_text.split("!")
                last_taggable.converters = converters

            last_taggable.substitution = sub_text

        elif c in ["<", "(", "[", "{", "|"]:
            # Begin group/tag/alt/etc.

            # Break literal here
            literal = literal.strip()
            if literal:
                assert last_group is not None
                words = list(split_words(literal))
                last_group.items.extend(words)

                last_word = words[-1]
                assert isinstance(last_word, Taggable)
                last_taggable = last_word
                literal = ""

            if c == "<":
                # Rule reference
                assert last_group is not None
                rule = RuleReference()
                end_index = parse_expression(
                    None, text[current_index + 1 :], end=[">"], is_literal=False
                )
                assert end_index
                next_index = end_index + current_index

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
                last_group.items.append(rule)
                last_taggable = rule
            elif c == "(":
                # Group (expression)
                assert last_group is not None
                group = Sequence(type=SequenceType.GROUP)
                end_index = parse_expression(
                    group, text[current_index + 1 :], end=[")"]
                )
                assert end_index
                next_index = end_index + current_index

                group = unwrap_sequence(group)
                group.text = text[current_index + 1 : next_index - 1]
                last_group.items.append(group)
                last_taggable = group
            elif c == "[":
                # Optional
                # Recurse with group sequence to capture multi-word children.
                optional_seq = Sequence(type=SequenceType.GROUP)
                end_index = parse_expression(
                    optional_seq, text[current_index + 1 :], end=["]"]
                )
                assert end_index
                next_index = end_index + current_index

                optional_seq = unwrap_sequence(optional_seq)
                optional = Sequence(type=SequenceType.ALTERNATIVE)
                if optional_seq.items:
                    if (
                        (len(optional_seq.items) == 1)
                        and (not optional_seq.tag)
                        and (not optional_seq.substitution)
                    ):
                        # Unpack inner item
                        # pylint: disable=E1136
                        inner_item = optional_seq.items[0]

                        # pylint: disable=E1101
                        optional.items.append(inner_item)
                    elif optional_seq.type == SequenceType.ALTERNATIVE:
                        # Unwrap inner alternative
                        # pylint: disable=E1101
                        optional.items.extend(optional_seq.items)
                    else:
                        # Keep inner group
                        optional_seq.text = text[current_index + 1 : next_index - 1]

                        # pylint: disable=E1101
                        optional.items.append(optional_seq)

                # Empty alternative
                # pylint: disable=E1101
                optional.items.append(Word(text=""))
                optional.text = text[current_index + 1 : next_index - 1]

                assert last_group is not None
                last_group.items.append(optional)
                last_taggable = optional
            elif c == "{":
                assert last_taggable is not None
                tag = Tag()

                # Tag
                end_index = parse_expression(
                    None, text[current_index + 1 :], end=["}"], is_literal=False
                )
                assert end_index
                next_index = end_index + current_index

                # Exclude {}
                tag.tag_text = text[current_index + 1 : next_index - 1]

                # Handle substitution/converter(s)
                if ":" in tag.tag_text or "!" in tag.tag_text:
                    if "!" in tag.tag_text:
                        # Word with converter(s)
                        # e.g., twenty:20!int
                        parts = tag.tag_text.split("!")
                        tag.tag_text = parts[0]
                        tag.converters = parts[1:]

                    if ":" in tag.tag_text:
                        # Word with substitution
                        # e.g., twenty:20
                        lhs, rhs = tag.tag_text.split(":", maxsplit=1)
                        tag.tag_text = lhs
                        tag.substitution = rhs

                last_taggable.tag = tag
            elif c == "|":
                assert root is not None
                if root.type != SequenceType.ALTERNATIVE:
                    # Create alternative
                    alternative = Sequence(type=SequenceType.ALTERNATIVE)
                    if len(root.items) == 1:
                        # Add directly
                        # pylint: disable=E1101
                        alternative.items.append(root.items[0])
                    else:
                        # Wrap in group
                        last_group = Sequence(type=SequenceType.GROUP, items=root.items)
                        # pylint: disable=E1101
                        alternative.items.append(last_group)

                    # Modify original sequence
                    root.items = [alternative]

                    # Overwrite root
                    root = alternative

                assert last_group is not None
                if not last_group.text:
                    # Fix text
                    last_group.text = " ".join(item.text for item in last_group.items)

                # Create new group for any follow-on expressions
                last_group = Sequence(type=SequenceType.GROUP)

                # pylint: disable=E1101
                alternative.items.append(last_group)
        else:
            # Accumulate into current literal
            literal += c

    # End of expression; Break literal.
    literal = literal.strip()
    if is_literal and literal:
        assert root is not None
        words = list(split_words(literal))
        assert last_group is not None
        last_group.items.extend(words)

    if last_group:
        if len(last_group.items) == 1:
            # Simplify final group
            assert root is not None
            root.items[-1] = last_group.items[0]
        elif not last_group.text:
            # Fix text
            last_group.text = " ".join(item.text for item in last_group.items)

    if end and (not found):
        # Signal end not found
        return None

    return next_index


# -----------------------------------------------------------------------------


def get_expression_count(
    expression: Expression,
    replacements: typing.Optional[typing.Dict[str, typing.Iterable[Sentence]]] = None,
    exclude_slots: bool = True,
) -> int:
    """Get the number of possible sentences in an expression."""
    if isinstance(expression, Sequence):
        if expression.type == SequenceType.GROUP:
            # Counts multiply down the sequence
            count = 1
            for sub_item in expression.items:
                count = count * get_expression_count(sub_item, replacements)

            return count

        if expression.type == SequenceType.ALTERNATIVE:
            # Counts sum across the alternatives
            return sum(
                get_expression_count(sub_item, replacements)
                for sub_item in expression.items
            )
    elif isinstance(expression, RuleReference):
        # Get substituted sentences for <rule>
        key = f"<{expression.rule_name}>"
        assert replacements, key
        return sum(
            get_expression_count(value, replacements) for value in replacements[key]
        )
    elif (not exclude_slots) and isinstance(expression, SlotReference):
        # Get substituted sentences for $slot
        key = f"${expression.slot_name}"
        assert replacements, key
        return sum(
            get_expression_count(value, replacements) for value in replacements[key]
        )
    elif isinstance(expression, Word):
        # Single word
        return 1

    return 0
