"""Test cases for ini/JSGF grammar parser."""
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

        intents = parse_ini(ini_text)
        self.assertEqual(
            intents,
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
                        items=[Word("this"), Word("is"), Word("another"), Word("test")],
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

        intents = parse_ini(ini_text)
        self.assertEqual(
            intents,
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

        intents = parse_ini(ini_text)
        intent_counts = get_intent_counts(intents)
        self.assertEqual(
            intent_counts,
            {
                "TestIntent1": (1 * 2 * 2 * 1) + (1 * 1 * 2 * 1),
                "TestIntent2": (1 * 1 * 3 * 1),
            },
        )

    def test_transform(self):
        """Test sentence transform."""
        ini_text = """
        [TestIntent1]
        THIS IS A TEST
        """

        intents = parse_ini(ini_text, sentence_transform=str.lower)
        self.assertEqual(
            intents,
            {
                "TestIntent1": [
                    Sentence(
                        text="this is a test",
                        items=[Word("this"), Word("is"), Word("a"), Word("test")],
                    )
                ]
            },
        )

    def test_intent_filter(self):
        """Test filtering intents."""
        ini_text = """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """

        intents = parse_ini(ini_text, intent_filter=lambda n: n != "TestIntent2")
        self.assertEqual(
            intents,
            {
                "TestIntent1": [
                    Sentence(
                        text="this is a test",
                        items=[Word("this"), Word("is"), Word("a"), Word("test")],
                    )
                ]
            },
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
