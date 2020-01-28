"""Types and constants."""
import typing

from .jsgf import Expression, Rule, Sentence

IntentsType = typing.Dict[str, typing.List[typing.Union[Sentence, Rule]]]
SentencesType = typing.Dict[str, typing.List[Sentence]]
ReplacementsType = typing.Dict[str, typing.List[Expression]]
