"""Grapheme to phoneme functions for word pronunciations."""
import logging
import re
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

PronunciationsType = typing.Dict[str, typing.List[typing.List[str]]]

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class MissingWordPronunciationsException(Exception):
    """Raised when missing word pronunciations and no g2p model."""

    def __init__(self, words: typing.List[str]):
        super().__init__(self)
        self.words = words

    def __str__(self):
        return f"Missing pronunciations for: {self.words}"


# -----------------------------------------------------------------------------


def read_pronunciations(
    dict_file: typing.Iterable[str],
    word_dict: typing.Optional[PronunciationsType] = None,
) -> PronunciationsType:
    """Loads a CMU-like pronunciation dictionary, optionally into an existing dictionary."""
    if word_dict is None:
        word_dict = {}

    for i, line in enumerate(dict_file):
        line = line.strip()
        if not line:
            continue

        try:
            # Use explicit whitespace (avoid 0xA0)
            word, *pronounce = re.split(r"[ \t]+", line)

            word = word.split("(")[0]

            if word in word_dict:
                word_dict[word].append(pronounce)
            else:
                word_dict[word] = [pronounce]
        except Exception as e:
            _LOGGER.warning("read_pronunciations: %s (line %s)", e, i + 1)

    return word_dict


def write_pronunciations(
    vocabulary: typing.Set[str],
    pronunciations: PronunciationsType,
    dictionary: typing.Union[str, Path],
    g2p_model: typing.Optional[typing.Union[str, Path]] = None,
    g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    phonetisaurus_apply: typing.Optional[typing.Union[str, Path]] = None,
    missing_words_path: typing.Optional[typing.Union[str, Path]] = None,
):
    """Create pronunciation dictionary. Guess missing words if g2p model is available."""
    # Look up words
    missing_words: typing.Set[str] = set()

    # Look up each word
    with open(dictionary, "w") as dictionary_file:
        for word in vocabulary:
            word_phonemes = pronunciations.get(word)
            if not word_phonemes:
                # Add to missing word list
                _LOGGER.warning("Missing word '%s'", word)
                missing_words.add(word)
                continue

            # Write CMU format
            for i, phonemes in enumerate(word_phonemes):
                phoneme_str = " ".join(phonemes).strip()
                if i == 0:
                    # word
                    print(word, phoneme_str, file=dictionary_file)
                else:
                    # word(n)
                    print(f"{word}({i+1})", phoneme_str, file=dictionary_file)

        # Open missing words file
        missing_file: typing.Optional[typing.TextIO] = None
        if missing_words_path:
            missing_file = open(missing_words_path, "w")

        try:
            if missing_words:
                # Fail if no g2p model is available
                if not g2p_model:
                    raise MissingWordPronunciationsException(list(missing_words))

                if not phonetisaurus_apply:
                    # Find in PATH
                    phonetisaurus_apply = shutil.which("phonetisaurus-apply")
                    assert phonetisaurus_apply, "phonetisaurus-apply not found in PATH"

                # Guess word pronunciations
                _LOGGER.debug("Guessing pronunciations for %s", missing_words)
                guesses = guess_pronunciations(
                    missing_words,
                    g2p_model,
                    phonetisaurus_apply,
                    g2p_word_transform=g2p_word_transform,
                    num_guesses=1,
                )

                # Output is a pronunciation dictionary.
                # Append to existing dictionary file.
                for guess_word, guess_phonemes in guesses:
                    guess_phoneme_str = " ".join(guess_phonemes).strip()
                    print(guess_word, guess_phoneme_str, file=dictionary_file)

                    if missing_file:
                        print(guess_word, guess_phoneme_str, file=missing_file)

        finally:
            if missing_file:
                missing_file.close()
                _LOGGER.debug("Wrote missing words to %s", str(missing_words_path))


def guess_pronunciations(
    words: typing.Iterable[str],
    g2p_model: typing.Union[str, Path],
    phonetisaurus_apply: typing.Optional[typing.Union[str, Path]] = None,
    g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    num_guesses: int = 1,
) -> typing.Iterable[typing.Tuple[str, typing.List[str]]]:
    """Guess phonetic pronunciations for words. Yields (word, phonemes) pairs."""
    if not phonetisaurus_apply:
        # Find in PATH
        phonetisaurus_apply = shutil.which("phonetisaurus-apply")
        assert phonetisaurus_apply, "phonetisaurus-apply not found in PATH"

    g2p_word_transform = g2p_word_transform or (lambda s: s)

    with tempfile.NamedTemporaryFile(mode="w") as wordlist_file:
        for word in words:
            word = g2p_word_transform(word)
            print(word, file=wordlist_file)

        wordlist_file.seek(0)
        g2p_command = [
            str(phonetisaurus_apply),
            "--model",
            str(g2p_model),
            "--word_list",
            wordlist_file.name,
            "--nbest",
            str(num_guesses),
        ]

        _LOGGER.debug(g2p_command)
        g2p_lines = subprocess.check_output(
            g2p_command, universal_newlines=True
        ).splitlines()

        # Output is a pronunciation dictionary.
        # Append to existing dictionary file.
        for line in g2p_lines:
            line = line.strip()
            if line:
                word, *phonemes = line.split()
                yield (word.strip(), phonemes)
