"""Test cases for ini/JSGF grammar parser."""
import io
import unittest

from ini_jsgf import IniJsgf
from jsgf import Sentence, Word, Sequence, SequenceType


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
            sentences = IniJsgf.parse(ini_file)
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
            sentences = IniJsgf.parse(ini_file)
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


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
