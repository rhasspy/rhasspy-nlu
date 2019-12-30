"""Test cases for recognition functions."""
import unittest
from pathlib import Path

from rhasspynlu.ini_jsgf import parse_ini
from rhasspynlu.jsgf_graph import intents_to_graph
from rhasspynlu.fsticuffs import recognize
from rhasspynlu.intent import Recognition, Intent, Entity
from rhasspynlu.jsgf import Sentence


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
        recognitions = zero_times(recognize("this is a test", graph, fuzzy=False))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Too many tokens (lower confidence)
        recognitions = zero_times(recognize("this is a bad test", graph, fuzzy=False))
        self.assertFalse(recognitions)

        # Too few tokens (failure)
        recognitions = zero_times(recognize("this is a", graph, fuzzy=False))
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
        recognitions = zero_times(recognize("this is a test", graph, fuzzy=False))
        self.assertEqual(len(recognitions), 2)
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent1", confidence=1),
                text="this is a test",
                raw_text="this is a test",
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent2", confidence=1),
                text="this is a test",
                raw_text="this is a test",
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
        recognitions = zero_times(recognize("this is a abcd test", graph, fuzzy=False))
        self.assertFalse(recognitions)

        # Success with stop words
        recognitions = zero_times(
            recognize(
                "this is a abcd test", graph, stop_words=set(["abcd"]), fuzzy=False
            )
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

    def test_converters(self):
        """Check sentence with converters."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test!c1
        """
        )

        graph = intents_to_graph(intents)

        # Recognition should still work
        recognitions = zero_times(recognize("this is a test", graph, fuzzy=False))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=[
                        "this",
                        "is",
                        "a",
                        "__begin_convert__c1",
                        "__begin_convert__c2",
                        "test",
                        "__end_convert__c1",
                        "__end_convert__c2",
                    ],
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
        recognitions = zero_times(recognize("this is a test", graph))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Too many tokens (lower confidence)
        recognitions = zero_times(recognize("this is a bad test", graph))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=(1 - 1 / 4)),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Too few tokens (failure)
        recognitions = zero_times(recognize("this is a", graph))
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
        recognitions = zero_times(recognize("this is a test", graph))
        self.assertEqual(len(recognitions), 2)
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent1", confidence=1),
                text="this is a test",
                raw_text="this is a test",
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )
        self.assertIn(
            Recognition(
                intent=Intent(name="TestIntent2", confidence=1),
                text="this is a test",
                raw_text="this is a test",
                tokens=["this", "is", "a", "test"],
                raw_tokens=["this", "is", "a", "test"],
            ),
            recognitions,
        )

    def test_intent_filter(self):
        """Identical sentences from two different intents with filter."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        def intent_filter(name):
            return name == "TestIntent1"

        # Should produce a recognition for first intent only
        recognitions = zero_times(
            recognize("this is a test", graph, intent_filter=intent_filter)
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent1", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
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
        recognitions = zero_times(recognize("this is a abcd test", graph))
        self.assertEqual(len(recognitions), 1)
        self.assertEqual(recognitions[0].intent.confidence, 1 - (1 / 4))

        # Higher confidence with stop words
        recognitions = zero_times(
            recognize("this is a abcd test", graph, stop_words=set(["abcd"]))
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=(1 - (0.1 / 4))),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

    def test_rules(self):
        """Make sure local and remote rules work."""
        intents = parse_ini(
            """
        [Intent1]
        rule = a test
        this is <rule>

        [Intent2]
        rule = this is
        <rule> <Intent1.rule>
        """
        )

        graph = intents_to_graph(intents)

        # Lower confidence with no stop words
        recognitions = zero_times(recognize("this is a test", graph))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="Intent1", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                ),
                Recognition(
                    intent=Intent(name="Intent2", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                ),
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
        recognitions = zero_times(
            recognize("set a timer for ten minutes", self.graph, fuzzy=False)
        )
        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        self.assertEqual(
            recognition.entities,
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


# -----------------------------------------------------------------------------


class MiscellaneousTestCase(unittest.TestCase):
    """Miscellaneous test cases."""

    def test_optional_entity(self):
        """Ensure entity inside optional is recognized."""
        ini_text = """
        [playBook]
        read me ($audio-book-name){book} in [the] [($assistant-zones){zone}]
        """

        replacements = {
            "$audio-book-name": [Sentence.parse("the hound of the baskervilles")],
            "$assistant-zones": [Sentence.parse("bedroom")],
        }

        graph = intents_to_graph(parse_ini(ini_text), replacements)

        recognitions = zero_times(
            recognize(
                "read me the hound of the baskervilles in the bedroom",
                graph,
                fuzzy=False,
            )
        )
        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        entities = {e.entity: e for e in recognition.entities}
        self.assertIn("book", entities)
        book = entities["book"]
        self.assertEqual(book.value, "the hound of the baskervilles")

        self.assertIn("zone", entities)
        zone = entities["zone"]
        self.assertEqual(zone.value, "bedroom")


# -----------------------------------------------------------------------------


def zero_times(recognitions):
    """Set times to zero so they can be easily compared in assertions"""
    for recognition in recognitions:
        recognition.recognize_seconds = 0

    return recognitions
