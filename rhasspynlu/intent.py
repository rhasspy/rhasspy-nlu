"""
Data structures for intent recognition.
"""
import typing
from enum import Enum

import attr


@attr.s(auto_attribs=True)
class Entity:
    """Named entity from intent."""

    entity: str
    value: str
    raw_value: str = ""
    start: int = 0
    raw_start: int = 0
    end: int = 0
    raw_end: int = 0
    tokens: typing.List[typing.Any] = attr.Factory(list)
    raw_tokens: typing.List[str] = attr.Factory(list)


@attr.s(auto_attribs=True)
class Intent:
    """Named intention with entities and slots."""

    name: str
    confidence: float = 0)


@attr.s(auto_attribs=True)
class TagInfo:
    """Information used to process FST tags."""

    tag: str
    start_index: int = 0
    raw_start_index: int = 0
    symbols: typing.List[str] = attr.Factory(list)
    raw_symbols: typing.List[str] = attr.Factory(list)


class RecognitionResult(str, Enum):
    """Result of a recognition."""

    SUCCESS = "success"
    FAILURE = "failure"


@attr.s(auto_attribs=True)
class Recognition:
    """Output of intent recognition."""

    intent: typing.Optional[Intent] = None
    entities: typing.List[Entity] = attr.Factory(list)
    text: str = ""
    raw_text: str = ""
    recognize_seconds: float = 0
    tokens: typing.List[typing.Any] = attr.Factory(list)
    raw_tokens: typing.List[str] = attr.Factory(list)

    def asdict(self) -> typing.Dict[str, typing.Any]:
        """Convert to dictionary."""
        return attr.asdict(self)
