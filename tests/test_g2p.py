"""Test cases for grapheme to phoneme."""
import io
import unittest

from rhasspynlu.g2p import load_sounds_like, load_g2p_corpus, read_pronunciations

_DICTIONARY = """
yawn Y AO N
test T EH S T
say S EY
who HH UW
bee B IY
azure AE ZH ER
read R EH D
read(2) R IY D
"""

_ALIGNMENT = """
a}AE z}ZH u|r}ER e}_
t}T e}EH s}S t}T
"""


class SoundsLikeTests(unittest.TestCase):
    """Test cases for sounds like pronunciations."""

    def setUp(self):
        """Set up tests"""
        with io.StringIO(_DICTIONARY) as dict_file:
            self.pronunciations = read_pronunciations(dict_file)

        with io.StringIO(_ALIGNMENT) as corpus_file:
            self.g2p_alignment = load_g2p_corpus(corpus_file)

    def test_known_words(self):
        """Test pronunciation from known words."""
        sounds_like = """
        beyoncé bee yawn say
        """

        with io.StringIO(sounds_like) as sounds_like_file:
            load_sounds_like(sounds_like_file, self.pronunciations)

        self.assertIn("beyoncé", self.pronunciations)

        # Verify pronunciation is the combination of known word phonemes
        phonemes = self.pronunciations["beyoncé"]
        self.assertEqual(len(phonemes), 1)
        self.assertEqual(phonemes[0], "B IY Y AO N S EY".split())

    def test_homonyms(self):
        """Test word with multiple pronunciations."""
        sounds_like = """
        readbee read bee
        """

        with io.StringIO(sounds_like) as sounds_like_file:
            load_sounds_like(sounds_like_file, self.pronunciations)

        self.assertIn("readbee", self.pronunciations)

        # Verify multiple pronunciations
        phonemes = self.pronunciations["readbee"]
        self.assertEqual(len(phonemes), 2)

        phoneme_strs = [" ".join(p) for p in phonemes]
        self.assertIn("R EH D B IY", phoneme_strs)
        self.assertIn("R IY D B IY", phoneme_strs)

    def test_choose_word(self):
        """Test choosing word from multiple pronunciations."""
        sounds_like = """
        readbee read(1) bee
        """

        with io.StringIO(sounds_like) as sounds_like_file:
            load_sounds_like(sounds_like_file, self.pronunciations)

        self.assertIn("readbee", self.pronunciations)

        # Verify single pronunciation
        phonemes = self.pronunciations["readbee"]
        self.assertEqual(len(phonemes), 1)
        self.assertEqual(phonemes[0], "R EH D B IY".split())

    def test_use_phonemes(self):
        """Test use of literal phonemes."""
        sounds_like = """
        hooiser who /ZH ER/
        """

        with io.StringIO(sounds_like) as sounds_like_file:
            load_sounds_like(sounds_like_file, self.pronunciations)

        self.assertIn("hooiser", self.pronunciations)

        # Verify combination of known word pronunciaton and literal phonemes
        phonemes = self.pronunciations["hooiser"]
        self.assertEqual(len(phonemes), 1)
        self.assertEqual(phonemes[0], "HH UW ZH ER".split())

    def test_word_segment(self):
        """Test use of word segment."""
        sounds_like = """
        hooiser who a>zure<
        """

        with io.StringIO(sounds_like) as sounds_like_file:
            load_sounds_like(
                sounds_like_file, self.pronunciations, g2p_alignment=self.g2p_alignment
            )

        self.assertIn("hooiser", self.pronunciations)

        # Verify combination of known word and word segment
        phonemes = self.pronunciations["hooiser"]
        self.assertEqual(len(phonemes), 1)
        self.assertEqual(phonemes[0], "HH UW ZH ER".split())

    def test_everything(self):
        """Test use of known words, literal phonemes, and word segments."""
        sounds_like = """
        hooiserreadboo who a>zure< read(2) /B UW/
        """

        with io.StringIO(sounds_like) as sounds_like_file:
            load_sounds_like(
                sounds_like_file, self.pronunciations, g2p_alignment=self.g2p_alignment
            )

        self.assertIn("hooiserreadboo", self.pronunciations)

        # Verify combination of known word, literal phonemes, and word segment
        phonemes = self.pronunciations["hooiserreadboo"]
        self.assertEqual(len(phonemes), 1)
        self.assertEqual(phonemes[0], "HH UW ZH ER R IY D B UW".split())


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
