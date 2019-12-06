"""Test cases for recognition functions."""
import unittest
from pathlib import Path

from .ini_jsgf import parse_ini
from .jsgf_graph import intents_to_graph
from .fsticuffs import recognize
from .intent import Recognition, Intent, Entity


class StrictTestCase(unittest.TestCase):
    """Strict recognition test cases."""

    def test_single_sentence(self):
        """Single intent, single sentence."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Exact
        recognitions = recognize("this is a test", graph, fuzzy=False)
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent"),
                    text="this is a test",
                    raw_text="this is a test",
                    confidence=1,
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Too many tokens (lower confidence)
        recognitions = recognize("this is a bad test", graph, fuzzy=False)
        self.assertFalse(recognitions)

        # Too few tokens (failure)
        recognitions = recognize("this is a", graph, fuzzy=False)
        self.assertFalse(recognitions)

    def test_multiple_sentences(self):
        """Identical sentences from two different intents."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Should produce a recognition for each intent
        recognitions = recognize("this is a test", graph, fuzzy=False)
        self.assertEqual(len(recognitions), 2)
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent1"),
                text="this is a test",
                raw_text="this is a test",
                confidence=1,
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent2"),
                text="this is a test",
                raw_text="this is a test",
                confidence=1,
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )

    def test_stop_words(self):
        """Check sentence with stop words."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Failure without stop words
        recognitions = recognize("this is a abcd test", graph, fuzzy=False)
        self.assertFalse(recognitions)

        # Success with stop words
        recognitions = recognize(
            "this is a abcd test", graph, stop_words=set(["abcd"]), fuzzy=False
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent"),
                    text="this is a test",
                    raw_text="this is a test",
                    confidence=1,
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )


# -----------------------------------------------------------------------------


class FuzzyTestCase(unittest.TestCase):
    """Fuzzy recognition test cases."""

    def test_single_sentence(self):
        """Single intent, single sentence."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Exact
        recognitions = recognize("this is a test", graph)
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent"),
                    text="this is a test",
                    raw_text="this is a test",
                    confidence=1,
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Too many tokens (lower confidence)
        recognitions = recognize("this is a bad test", graph)
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent"),
                    text="this is a test",
                    raw_text="this is a test",
                    confidence=(1 - 1 / 4),
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Too few tokens (failure)
        recognitions = recognize("this is a", graph)
        self.assertFalse(recognitions)

    def test_multiple_sentences(self):
        """Identical sentences from two different intents."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Should produce a recognition for each intent
        recognitions = recognize("this is a test", graph)
        self.assertEqual(len(recognitions), 2)
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent1"),
                text="this is a test",
                raw_text="this is a test",
                confidence=1,
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent2"),
                text="this is a test",
                raw_text="this is a test",
                confidence=1,
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )

    def test_stop_words(self):
        """Check sentence with stop words."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Lower confidence with no stop words
        recognitions = recognize("this is a abcd test", graph)
        self.assertEqual(len(recognitions), 1)
        self.assertEqual(recognitions[0].confidence, 1 - (1 / 4))

        # Higher confidence with stop words
        recognitions = recognize("this is a abcd test", graph, stop_words=set(["abcd"]))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent"),
                    text="this is a test",
                    raw_text="this is a test",
                    confidence=(1 - (0.1 / 4)),
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )


# -----------------------------------------------------------------------------


class TimerTestCase(unittest.TestCase):
    """Test cases for timer example."""

    def setUp(self):
        # Load timer example
        self.graph = intents_to_graph(parse_ini(Path("etc") / "timer.ini"))

    def test_strict_simple(self):
        """Check exact parse."""
        recognitions = recognize("set a timer for ten minutes", self.graph, fuzzy=False)
        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        self.assertEqual(
            recognition.intent.entities,
            [
                Entity(
                    entity="minutes",
                    value="10",
                    raw_value="ten",
                    start=16,
                    raw_start=16,
                    raw_end=19,
                    end=18,
                    tokens=["10"],
                    raw_tokens=["ten"],
                )
            ],
        )
