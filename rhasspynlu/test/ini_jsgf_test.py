"""Test cases for ini/JSGF grammar parser."""
import io
import unittest

from rhasspynlu.ini_jsgf import parse_ini, get_intent_counts
from rhasspynlu.jsgf import Sentence, Word, Sequence, SequenceType


class IniJsgfTestCase(unittest.TestCase):
    """Test cases for ini/JSGF grammar parser."""

    def test_parse(self):
        """Test ini/JSGF parser."""
        ini_text = """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """

        with io.StringIO(ini_text) as ini_file:
            sentences = parse_ini(ini_file)
            self.assertEqual(
                sentences,
                {
                    "TestIntent1": [
                        Sentence(
                            text="this is a test",
                            items=[Word("this"), Word("is"), Word("a"), Word("test")],
                        )
                    ],
                    "TestIntent2": [
                        Sentence(
                            text="this is another test",
                            items=[
                                Word("this"),
                                Word("is"),
                                Word("another"),
                                Word("test"),
                            ],
                        )
                    ],
                },
            )

    def test_escape(self):
        """Test escaped optional."""
        ini_text = """
        [TestIntent1]
        \\[this] is a test
        """

        with io.StringIO(ini_text) as ini_file:
            sentences = parse_ini(ini_file)
            self.assertEqual(
                sentences,
                {
                    "TestIntent1": [
                        Sentence(
                            text="[this] is a test",
                            items=[
                                Sequence(
                                    text="this",
                                    type=SequenceType.ALTERNATIVE,
                                    items=[Word("this"), Word("")],
                                ),
                                Word("is"),
                                Word("a"),
                                Word("test"),
                            ],
                        )
                    ]
                },
            )

    def test_intent_counts(self):
        """Test sentence counts by intent."""
        ini_text = """
        [TestIntent1]
        this [is] [a] test
        this is [another] test

        [TestIntent2]
        this is (my | your| another) test
        """

        with io.StringIO(ini_text) as ini_file:
            intents = parse_ini(ini_file)
            intent_counts = get_intent_counts(intents)
            self.assertEqual(
                intent_counts,
                {
                    "TestIntent1": (1 * 2 * 2 * 1) + (1 * 1 * 2 * 1),
                    "TestIntent2": (1 * 1 * 3 * 1),
                },
            )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
