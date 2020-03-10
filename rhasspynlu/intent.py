"""
Data structures for intent recognition.
"""
import typing
from enum import Enum

import attr


@attr.s(auto_attribs=True, slots=True)
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

    @classmethod
    def from_dict(cls, entity_dict: typing.Dict[str, typing.Any]) -> "Entity":
        """Create Entity from dictionary."""
        return Entity(**entity_dict)


@attr.s(auto_attribs=True, slots=True)
class Intent:
    """Named intention with entities and slots."""

    name: str
    confidence: float = 0

    @classmethod
    def from_dict(cls, intent_dict: typing.Dict[str, typing.Any]) -> "Intent":
        """Create Intent from dictionary."""
        return Intent(**intent_dict)


@attr.s(auto_attribs=True, slots=True)
class TagInfo:
    """Information used to process FST tags."""

    tag: str
    start_index: int = 0
    raw_start_index: int = 0
    symbols: typing.List[str] = attr.Factory(list)
    raw_symbols: typing.List[str] = attr.Factory(list)

    @classmethod
    def from_dict(cls, tag_dict: typing.Dict[str, typing.Any]) -> "TagInfo":
        """Create TagInfo from dictionary."""
        return TagInfo(**tag_dict)


class RecognitionResult(str, Enum):
    """Result of a recognition."""

    SUCCESS = "success"
    FAILURE = "failure"


@attr.s(auto_attribs=True, slots=True)
class Recognition:
    """Output of intent recognition."""

    intent: typing.Optional[Intent] = None
    entities: typing.List[Entity] = attr.Factory(list)
    text: str = ""
    raw_text: str = ""
    recognize_seconds: float = 0
    tokens: typing.List[typing.Any] = attr.Factory(list)
    raw_tokens: typing.List[str] = attr.Factory(list)

    # Transcription details
    wav_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    speech_confidence: float = 0.0

    def asdict(self) -> typing.Dict[str, typing.Any]:
        """Convert to dictionary."""
        return attr.asdict(self)

    @classmethod
    def empty(cls) -> "Recognition":
        """Return an empty recognition."""
        return Recognition(intent=Intent(name=""))

    @classmethod
    def from_dict(cls, recognition_dict: typing.Dict[str, typing.Any]) -> "Recognition":
        """Create Recognition from dictionary."""
        # Exclude unused fields from Rhasspy JSON format
        recognition_dict.pop("intents", None)

        intent_dict = recognition_dict.pop("intent", None)
        entity_dicts = recognition_dict.pop("entities", None)
        slots_dict = recognition_dict.pop("slots", None)
        recognition = Recognition(**recognition_dict)

        if intent_dict:
            recognition.intent = Intent.from_dict(intent_dict)

        if entity_dicts:
            recognition.entities = [Entity.from_dict(e) for e in entity_dicts]

        if slots_dict:
            recognition.entities = [
                Entity(entity=key, value=value) for key, value in slots_dict.items()
            ]

        return recognition
