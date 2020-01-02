"""
Data structures for intent recognition.
"""
from enum import Enum
import typing

import attr


@attr.s
class Entity:
    """Named entity from intent."""

    entity: str = attr.ib()
    value: str = attr.ib()
    raw_value: str = attr.ib(default="")
    start: int = attr.ib(default=0)
    raw_start: int = attr.ib(default=0)
    end: int = attr.ib(default=0)
    raw_end: int = attr.ib(default=0)
    tokens: typing.List[typing.Any] = attr.ib(factory=list)
    raw_tokens: typing.List[str] = attr.ib(factory=list)


@attr.s
class Intent:
    """Named intention with entities and slots."""

    name: str = attr.ib()
    confidence: float = attr.ib(default=0)


@attr.s
class TagInfo:
    """Information used to process FST tags."""

    tag: str = attr.ib()
    start_index: int = attr.ib(default=0)
    raw_start_index: int = attr.ib(default=0)
    symbols: typing.List[str] = attr.ib(factory=list)
    raw_symbols: typing.List[str] = attr.ib(factory=list)


class RecognitionResult(str, Enum):
    """Result of a recognition."""

    SUCCESS = "success"
    FAILURE = "failure"


@attr.s
class Recognition:
    """Output of intent recognition."""

    intent: typing.Optional[Intent] = attr.ib(default=None)
    entities: typing.List[Entity] = attr.ib(factory=list)
    text: str = attr.ib(default="")
    raw_text: str = attr.ib(default="")
    recognize_seconds: float = attr.ib(default=0)
    tokens: typing.List[typing.Any] = attr.ib(factory=list)
    raw_tokens: typing.List[str] = attr.ib(factory=list)

    def asdict(self) -> typing.Dict[str, typing.Any]:
        """Convert to dictionary."""
        return attr.asdict(self)
