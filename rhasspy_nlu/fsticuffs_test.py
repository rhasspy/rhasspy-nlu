"""Test cases for recognition functions."""
import unittest

from .ini_jsgf import parse_ini
from .jsgf_graph import intents_to_graph
from .fsticuffs import recognize_strict, recognize_fuzzy, best_fuzzy_cost


class StrictTestCase(unittest.TestCase):
    """Strict recognition test cases."""

    def test_single_sentence(self):
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Exact
        tokens = ["this", "is", "a", "test"]
        result = recognize_strict(tokens, graph)
        self.assertEqual(result, ["__label__TestIntent"] + tokens)

        # Too many tokens
        tokens = ["this", "is", "a", "bad", "test"]
        result = recognize_strict(tokens, graph)
        self.assertFalse(result)

        # Too few tokens
        tokens = ["this", "is", "a"]
        result = recognize_strict(tokens, graph)
        self.assertFalse(result)

    def test_multiple_sentences(self):
        intents = parse_ini(
            """
        [TestIntent1]
        this is a first test
        this is a second test

        [TestIntent2]
        this is a third test
        """
        )

        graph = intents_to_graph(intents)

        # TestIntent1
        tokens = ["this", "is", "a", "first", "test"]
        result = recognize_strict(tokens, graph)
        self.assertEqual(result, ["__label__TestIntent1"] + tokens)

        # TestIntent2
        tokens = ["this", "is", "a", "third", "test"]
        result = recognize_strict(tokens, graph)
        self.assertEqual(result, ["__label__TestIntent2"] + tokens)


# -----------------------------------------------------------------------------


class FuzzyTestCase(unittest.TestCase):
    """Fuzzy recognition test cases."""

    def test_single_sentence(self):
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)

        # Exact
        tokens = ["this", "is", "a", "test"]
        results = best_fuzzy_cost(recognize_fuzzy(tokens, graph))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens, ["__label__TestIntent"] + tokens)
        self.assertEqual(results[0].cost, 0)

        # Too many tokens
        tokens = ["this", "is", "a", "bad", "test"]
        results = best_fuzzy_cost(recognize_fuzzy(tokens, graph))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens, ["__label__TestIntent", "this", "is", "a", "test"])
        self.assertGreater(results[0].cost, 0)

        # Too few tokens
        tokens = ["this", "is", "a"]
        results = best_fuzzy_cost(recognize_fuzzy(tokens, graph))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens, ["__label__TestIntent", "this", "is", "a", "test"])
        self.assertGreater(results[0].cost, 0)
