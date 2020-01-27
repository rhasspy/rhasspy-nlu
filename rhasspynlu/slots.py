"""Slot load/parsing utility methods."""
import logging
import subprocess
import typing
from pathlib import Path

import attr

from .jsgf import Expression, Rule, Sentence, Sequence, SlotReference, walk_expression

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s
class StaticSlotInfo:
    """Name/path to a static slot text file."""

    name: str = attr.ib()
    path: Path = attr.ib()


@attr.s
class SlotProgramInfo:
    """Name/path/arguments for a slot program."""

    key: str = attr.ib()
    name: str = attr.ib()
    path: Path = attr.ib()
    args: typing.Optional[typing.List[str]] = attr.ib(default=None)


# -----------------------------------------------------------------------------


def get_slot_replacements(
    intents,
    slots_dirs: typing.List[Path],
    slot_programs_dirs: typing.List[Path],
    word_transform: typing.Optional[
        typing.Callable[[Expression], typing.Union[bool, typing.Optional[Expression]]]
    ] = None,
) -> typing.Dict[str, typing.Iterable[Sentence]]:
    """Create replacement dictionary for referenced slots."""
    replacements: typing.Dict[str, typing.Iterable[Sentence]] = {}

    # Gather used slot names
    slot_names: typing.Set[str] = set()
    for intent_name in intents:
        for item in intents[intent_name]:
            for slot_name in get_slot_names(item):
                slot_names.add(slot_name)

    # Load slot values
    for slot_key in slot_names:
        if slot_key in replacements:
            # Skip already loaded slot
            continue

        # Find slot file/program in file system
        slot_info = find_slot(slot_key, slots_dirs, slot_programs_dirs)

        if isinstance(slot_info, StaticSlotInfo):
            # Parse each non-empty line as a JSGF sentence
            slot_values = []
            _LOGGER.debug("Loading slot %s from %s", slot_key, str(slot_info.path))
            with open(slot_info.path, "r") as slot_file:
                for line in slot_file:
                    line = line.strip()
                    if line:
                        sentence = Sentence.parse(line)
                        if word_transform:
                            walk_expression(sentence, word_transform)

                        slot_values.append(sentence)
        elif isinstance(slot_info, SlotProgramInfo):
            # Generate values in place
            slot_command = [str(slot_info.path)] + (slot_info.args or [])
            _LOGGER.debug("Running program for slot %s: %s", slot_key, slot_command)

            # Parse each non-empty line as a JSGF sentence
            slot_values = []
            for line in subprocess.check_output(
                slot_command, universal_newlines=True
            ).splitlines():
                line = line.strip()
                if line:
                    sentence = Sentence.parse(line)
                    if word_transform:
                        walk_expression(sentence, word_transform)

                    slot_values.append(sentence)
        else:
            _LOGGER.warning(
                "Failed to load file/program for slot %s (tried: %s, %s)",
                slot_key,
                slots_dirs,
                slot_programs_dirs,
            )
            slot_values = []

        # Replace $slot with sentences
        replacements[f"${slot_key}"] = slot_values

    return replacements


# -----------------------------------------------------------------------------


def get_slot_names(item: Expression) -> typing.Iterable[str]:
    """Yield referenced slot names from an expression."""
    if isinstance(item, SlotReference):
        yield item.slot_name
    elif isinstance(item, Sequence):
        for sub_item in item.items:
            for slot_name in get_slot_names(sub_item):
                yield slot_name
    elif isinstance(item, Rule):
        for slot_name in get_slot_names(item.rule_body):
            yield slot_name


def split_slot_args(
    slot_name: str
) -> typing.Tuple[str, typing.Optional[typing.List[str]]]:
    """Split slot name and arguments out (slot,arg1,arg2,...)"""
    # Check for arguments.
    slot_args: typing.Optional[typing.List[str]] = None

    # Slot name retains argument(s).
    if "," in slot_name:
        slot_name, *slot_args = slot_name.split(",")

    return slot_name, slot_args


# -----------------------------------------------------------------------------


def find_slot(
    slot_key: str, slots_dirs: typing.List[Path], slot_programs_dirs: typing.List[Path]
) -> typing.Optional[typing.Union[StaticSlotInfo, SlotProgramInfo]]:
    """Look up a static slot or slot program."""
    # Try static user slots
    for slots_dir in slots_dirs:
        slot_path = slots_dir / slot_key
        if slot_path.is_file():
            return StaticSlotInfo(name=slot_key, path=slot_path)

    # Try user slot programs
    slot_name, slot_args = split_slot_args(slot_key)
    for slot_programs_dir in slot_programs_dirs:
        slot_path = slot_programs_dir / slot_name
        if slot_path.is_file():
            return SlotProgramInfo(
                key=slot_key, name=slot_name, path=slot_path, args=slot_args
            )

    return None
